"""
Material Simulation Dashboard (Streamlit)
----------------------------------------
Goal: A professional, Materials-Studio-like mini environment for building,
visualizing, analyzing, and exporting structures.

Stack:
 - UI: Streamlit (custom dark theme CSS)
 - 3D viewer: py3Dmol embedded via Streamlit components
 - Chemistry/structures: pymatgen (+ RDKit for bond inference where possible)
 - Physical constants: scipy.constants (authoritative CODATA values)
 - Data: numpy, pandas

Memory note (6GB RAM target):
 - Avoid huge supercells by default; cache generated structures.
 - Keep only current structure in session_state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st

import py3Dmol
from stmol import showmol
from scipy import constants as sc

from pymatgen.core import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter

# RDKit is used (when available) to infer bonds from XYZ reliably for molecules.
# On some systems, RDKit installation can be tricky; we keep a safe fallback path.
try:
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    _HAVE_RDKIT = True
except Exception:
    _HAVE_RDKIT = False


# -----------------------------
# UI: Dark scientific dashboard
# -----------------------------

_DARK_CSS = """
<style>
  /* Global app background */
  .stApp {
    background: radial-gradient(1200px 900px at 20% 10%, #1a2234 0%, #0b1020 55%, #070b15 100%);
    color: #e7eefc;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1428 0%, #0a1020 100%);
    border-right: 1px solid rgba(120, 160, 255, 0.14);
  }

  /* Headings */
  h1, h2, h3 {
    letter-spacing: 0.2px;
    color: #f4f7ff;
  }

  /* Cards (containers) */
  div[data-testid="stVerticalBlock"] div[data-testid="stContainer"] {
    border: 1px solid rgba(120, 160, 255, 0.16);
    background: rgba(10, 16, 32, 0.68);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
  }

  /* Buttons */
  .stButton>button {
    background: linear-gradient(180deg, rgba(64,130,255,0.95) 0%, rgba(46,102,220,0.95) 100%);
    border: 1px solid rgba(160, 200, 255, 0.22);
    color: white;
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
    transition: transform 0.06s ease-in-out;
  }
  .stButton>button:active { transform: scale(0.99); }

  /* Inputs */
  input, textarea {
    background: rgba(8, 12, 24, 0.65) !important;
    color: #e7eefc !important;
    border: 1px solid rgba(120, 160, 255, 0.18) !important;
    border-radius: 10px !important;
  }

  /* Dataframe */
  div[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(120, 160, 255, 0.16);
  }

  /* Subtle glow separators */
  hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(120,160,255,0.25), transparent);
    margin: 0.8rem 0;
  }

  /* Make markdown links nicer */
  a { color: #93c5fd; }
</style>
"""


def _inject_css() -> None:
    st.markdown(_DARK_CSS, unsafe_allow_html=True)


# -----------------------------
# Domain models & constants
# -----------------------------


@dataclass(frozen=True)
class BuildResult:
    name: str
    structure: Structure
    is_molecule_like: bool  # True for non-periodic / pseudo-periodic constructs (e.g., CNT)
    metadata: Dict[str, Union[str, float, int]]


# Graphene geometry
_A_CC_ANG = 1.42  # Å  (C–C bond length)
_A_GRAPHENE_ANG = 2.46  # Å  (graphene lattice constant, ~sqrt(3)*a_cc)


# -----------------------------
# Structure builders (pymatgen)
# -----------------------------


def _build_diamond(a_ang: float = 3.567) -> Structure:
    """
    Diamond cubic (minimal conventional representation).
    Note: This is a compact unit cell (2 atoms). For visualization, supercells help.
    """
    # Diamond basis (fractional): (0,0,0) and (1/4,1/4,1/4)
    lat = Lattice.cubic(a_ang)
    species = ["C", "C"]
    frac = [[0, 0, 0], [0.25, 0.25, 0.25]]
    return Structure(lat, species, frac)


def _build_silicon(a_ang: float = 5.431) -> Structure:
    """Silicon diamond structure (same basis as diamond, different lattice parameter)."""
    lat = Lattice.cubic(a_ang)
    species = ["Si", "Si"]
    frac = [[0, 0, 0], [0.25, 0.25, 0.25]]
    return Structure(lat, species, frac)


def _build_gold(a_ang: float = 4.078) -> Structure:
    """Gold fcc (primitive 1-atom cell)."""
    lat = Lattice.cubic(a_ang)
    species = ["Au"]
    frac = [[0, 0, 0]]
    return Structure(lat, species, frac)


def _build_graphene(a_ang: float = _A_GRAPHENE_ANG, vacuum_c_ang: float = 20.0) -> Structure:
    """
    Graphene (2D sheet) represented as periodic in a,b and large vacuum in c.
    Primitive hex cell with two atoms.
    """
    lat = Lattice.hexagonal(a_ang, vacuum_c_ang)
    # Basis (fractional) for graphene in hex lattice:
    # (0,0,0) and (1/3,2/3,0)
    species = ["C", "C"]
    frac = [[0, 0, 0], [1 / 3, 2 / 3, 0]]
    return Structure(lat, species, frac)


def _gcd_int(a: int, b: int) -> int:
    a = abs(int(a))
    b = abs(int(b))
    while b:
        a, b = b, a % b
    return max(a, 1)


def _build_cnt_chiral(n: int, m: int, length_ang: float) -> BuildResult:
    """
    Build a carbon nanotube (CNT) using the chiral vector method.

    Scientific background:
    - Graphene lattice vectors (2D) using lattice constant a ≈ 2.46 Å:
        a1 = (a/2)(√3,  1)
        a2 = (a/2)(√3, -1)
    - Chiral vector:
        C = n a1 + m a2    (circumference direction)
      |C| is the tube circumference => radius r = |C| / (2π)
    - Translational vector along tube axis:
        d_R = gcd(2m+n, 2n+m)
        t1 = (2m+n)/d_R,   t2 = -(2n+m)/d_R
        T = t1 a1 + t2 a2
      |T| is the CNT unit cell length along z.

    Arabic/English notes (equations):
    - EN: Tube radius from circumference: r = |C|/(2π)
      AR: نصف قطر الأنبوب من المحيط: \( r = |C|/(2\pi) \)
    """
    if n <= 0 or m < 0:
        raise ValueError("CNT chirality must satisfy n>0 and m>=0.")
    if length_ang <= 0:
        raise ValueError("CNT length must be > 0 Å.")

    a = float(_A_GRAPHENE_ANG)

    # 2D lattice vectors (Å)
    a1 = np.array([math.sqrt(3) * a / 2.0, a / 2.0], dtype=float)
    a2 = np.array([math.sqrt(3) * a / 2.0, -a / 2.0], dtype=float)

    C = n * a1 + m * a2
    C_len = float(np.linalg.norm(C))
    r = C_len / (2.0 * math.pi)

    dR = _gcd_int(2 * m + n, 2 * n + m)
    t1 = (2 * m + n) // dR
    t2 = -(2 * n + m) // dR
    T = t1 * a1 + t2 * a2
    T_len = float(np.linalg.norm(T))

    # Primitive basis for graphene (Å)
    # b1=(0,0), b2=(a/√3, 0) ≈ (1.42, 0)
    b1 = np.array([0.0, 0.0], dtype=float)
    b2 = np.array([a / math.sqrt(3), 0.0], dtype=float)

    # Build a 2D unit cell parallelogram spanned by C and T.
    # We'll enumerate a bounded integer grid of lattice points and filter by (u,v) in [0,1).
    M = np.column_stack([C, T])  # 2x2
    Minv = np.linalg.inv(M)

    # Choose scan bounds proportional to |C| and |T| to ensure coverage.
    # This is conservative but still small for typical (n,m).
    scan = int(max(6, n + m + 6))
    points_2d = []
    uvs = []

    for i in range(-scan, scan + 1):
        for j in range(-scan, scan + 1):
            R = i * a1 + j * a2
            for basis in (b1, b2):
                p = R + basis
                uv = Minv @ p  # coefficients in (C,T) basis
                u, v = float(uv[0]), float(uv[1])
                if 0.0 <= u < 1.0 and 0.0 <= v < 1.0:
                    points_2d.append(p)
                    uvs.append((u, v))

    if not points_2d:
        raise RuntimeError("Failed to generate CNT unit cell (no atoms found).")

    # Deduplicate: numerical rounding tolerance.
    # We dedupe in (u,v) space since rolling maps u->theta.
    uv_arr = np.array(uvs, dtype=float)
    uv_round = np.round(uv_arr, 6)
    _, uniq_idx = np.unique(uv_round, axis=0, return_index=True)

    uv_arr = uv_arr[uniq_idx]

    # Replicate along tube axis to reach requested length.
    # EN: n_cells = ceil(L / |T|)
    # AR: عدد الخلايا على طول الأنبوب = ceil(الطول / |T|)
    n_cells = int(max(1, math.ceil(length_ang / T_len)))

    # Build 3D coordinates by rolling:
    # theta = 2πu, z = (v + cell)*|T|, radius r
    coords = []
    for cell in range(n_cells):
        z0 = cell * T_len
        for u, v in uv_arr:
            theta = 2.0 * math.pi * float(u)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            z = z0 + float(v) * T_len
            coords.append((x, y, z))

    coords = np.array(coords, dtype=float)

    # Create a pseudo-periodic Structure to enable CIF export.
    # We provide a generous vacuum in x,y so atoms don't overlap across boundaries.
    pad = 12.0
    diam = 2.0 * r
    a_box = float(diam + 2 * pad)
    b_box = float(diam + 2 * pad)
    c_box = float(n_cells * T_len + 2 * pad)

    # Shift to center the molecule in the box (cartesian).
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    target_center = np.array([a_box / 2.0, b_box / 2.0, c_box / 2.0], dtype=float)
    coords_centered = coords - center + target_center

    lat = Lattice.from_parameters(a_box, b_box, c_box, 90, 90, 90)
    struct = Structure(lat, ["C"] * len(coords_centered), coords_centered, coords_are_cartesian=True)

    meta = {
        "n": int(n),
        "m": int(m),
        "tube_radius_ang": float(r),
        "tube_diameter_ang": float(2.0 * r),
        "unit_cell_T_ang": float(T_len),
        "cells": int(n_cells),
        "requested_length_ang": float(length_ang),
        "actual_length_ang": float(n_cells * T_len),
    }
    return BuildResult(name=f"CNT ({n},{m})", structure=struct, is_molecule_like=True, metadata=meta)


@st.cache_data(show_spinner=False)
def build_structure_cached(
    preset: str,
    lattice_a: float,
    graphene_vac_c: float,
    supercell: Tuple[int, int, int],
    cnt_n: int,
    cnt_m: int,
    cnt_length: float,
) -> BuildResult:
    """
    Cached builder: keeps memory low by caching only small structures
    and rebuilding when parameters change.
    """
    if preset == "Diamond":
        s = _build_diamond(a_ang=lattice_a)
        name = "Diamond (C)"
        is_mol = False
        meta: Dict[str, Union[str, float, int]] = {"a_ang": float(lattice_a)}
    elif preset == "Gold":
        s = _build_gold(a_ang=lattice_a)
        name = "Gold (Au, fcc)"
        is_mol = False
        meta = {"a_ang": float(lattice_a)}
    elif preset == "Silicon":
        s = _build_silicon(a_ang=lattice_a)
        name = "Silicon (Si)"
        is_mol = False
        meta = {"a_ang": float(lattice_a)}
    elif preset == "Graphene":
        s = _build_graphene(a_ang=lattice_a, vacuum_c_ang=graphene_vac_c)
        name = "Graphene (C, 2D)"
        is_mol = False
        meta = {"a_ang": float(lattice_a), "vacuum_c_ang": float(graphene_vac_c)}
    elif preset == "CNT":
        res = _build_cnt_chiral(cnt_n, cnt_m, cnt_length)
        s = res.structure
        name = res.name
        is_mol = res.is_molecule_like
        meta = dict(res.metadata)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    # Small supercells help visuals but can blow up memory quickly.
    # EN: Supercell atoms scale ~ nx*ny*nz
    # AR: عدد الذرات في السوبرسيل يتناسب تقريباً مع nx*ny*nz
    if preset != "CNT":
        nx, ny, nz = supercell
        if (nx, ny, nz) != (1, 1, 1):
            s = s.copy()
            s.make_supercell([nx, ny, nz])
            meta["supercell"] = f"{nx}x{ny}x{nz}"

    return BuildResult(name=name, structure=s, is_molecule_like=is_mol, metadata=meta)


# -----------------------------
# Analysis (constants-accurate)
# -----------------------------


def _structure_mass_kg(struct: Structure) -> float:
    """
    Mass of all atoms in the structure in kg.

    EN: mass = Σ_i (atomic_mass_i [amu] * m_u)
    AR: الكتلة = مجموع (الكتلة الذرية [amu] × ثابت الكتلة الذرية m_u)
    where m_u = scipy.constants.atomic_mass (kg).
    """
    m_u = sc.atomic_mass  # kg / amu
    total_amu = 0.0
    for sp in struct.species:
        # pymatgen species can be Element or Specie; Element(sp.symbol) is safe.
        el = Element(str(sp))
        total_amu += float(el.atomic_mass)
    return float(total_amu * m_u)


def _density_g_cm3(struct: Structure) -> Optional[float]:
    """
    Density for periodic structures (g/cm^3).
    For pseudo-molecular CNT in a big box, this is not physically meaningful,
    so we return None and compute a CNT-specific density separately.
    """
    # If the lattice contains huge vacuum, density becomes artificially small.
    # We detect "molecule-like" by heuristic: very large a,b relative to nearest distance.
    vol_ang3 = float(struct.lattice.volume)  # Å^3
    if vol_ang3 <= 0:
        return None

    mass_kg = _structure_mass_kg(struct)
    vol_m3 = vol_ang3 * 1e-30  # (1 Å = 1e-10 m) => Å^3 = 1e-30 m^3
    rho_kg_m3 = mass_kg / vol_m3
    rho_g_cm3 = rho_kg_m3 / 1000.0  # 1 g/cm^3 = 1000 kg/m^3
    return float(rho_g_cm3)


def _count_bonds(struct: Structure, is_molecule_like: bool) -> int:
    """
    Count bonds approximately.

    Strategy:
    - If molecule-like and RDKit available: infer bonds from XYZ and count.
    - Else: distance cutoff heuristic.

    EN: Bond if distance < cutoff (Å)
    AR: نعتبر رابطاً إذا كانت المسافة < حد القطع (Å)
    """
    coords = np.array(struct.cart_coords, dtype=float)
    species = [str(s) for s in struct.species]

    if is_molecule_like and _HAVE_RDKIT:
        xyz = xyz_from_structure(struct, comment="rdkit_bond_infer")
        mol = Chem.MolFromXYZBlock(xyz)
        if mol is not None:
            try:
                rdDetermineBonds.DetermineBonds(mol)
                return int(mol.GetNumBonds())
            except Exception:
                pass  # fallback below

    # Fallback: cutoff based on element pairs (simple, robust, fast)
    # Default cutoffs (Å) for common bonds; otherwise use covalent radii sum * factor.
    pt = None
    if _HAVE_RDKIT:
        try:
            pt = Chem.GetPeriodicTable()
        except Exception:
            pt = None

    n = len(coords)
    if n <= 1:
        return 0

    # O(N^2) is OK for the small structures we generate (we clamp supercells).
    bonds = 0
    for i in range(n):
        ei = species[i]
        for j in range(i + 1, n):
            ej = species[j]
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if (ei == "C" and ej == "C") or (ei == "Si" and ej == "Si"):
                cutoff = 1.85  # covers sp2/sp3 and Si–Si
            elif (ei == "Au" and ej == "Au"):
                cutoff = 3.2
            else:
                if pt is not None:
                    ri = float(pt.GetRcovalent(pt.GetAtomicNumber(ei)))
                    rj = float(pt.GetRcovalent(pt.GetAtomicNumber(ej)))
                    cutoff = 1.25 * (ri + rj)
                else:
                    cutoff = 2.0
            if d <= cutoff and d > 0.4:
                bonds += 1
    return int(bonds)


def _cnt_solid_cylinder_density_g_cm3(struct: Structure, cnt_meta: Dict[str, Union[str, float, int]]) -> Optional[float]:
    """
    CNT "apparent" density assuming a solid cylinder of radius r and length L.
    This is NOT the true material density (CNT is hollow), but is a useful scalar.

    EN: ρ = mass / (π r^2 L)
    AR: الكثافة التقريبية = الكتلة / (π r^2 L)

    Units:
    - r, L in Å -> convert to meters.
    """
    try:
        r_ang = float(cnt_meta["tube_radius_ang"])
        L_ang = float(cnt_meta["actual_length_ang"])
    except Exception:
        return None
    if r_ang <= 0 or L_ang <= 0:
        return None

    mass_kg = _structure_mass_kg(struct)
    r_m = r_ang * 1e-10
    L_m = L_ang * 1e-10
    vol_m3 = math.pi * (r_m**2) * L_m
    rho_g_cm3 = (mass_kg / vol_m3) / 1000.0
    return float(rho_g_cm3)


def summarize_properties(res: BuildResult) -> pd.DataFrame:
    s = res.structure
    natoms = int(len(s))

    a, b, c = map(float, s.lattice.abc)
    alpha, beta, gamma = map(float, s.lattice.angles)
    vol = float(s.lattice.volume)

    # Density
    rho = _density_g_cm3(s)
    if res.is_molecule_like and "tube_radius_ang" in res.metadata:
        rho_cnt = _cnt_solid_cylinder_density_g_cm3(s, res.metadata)
    else:
        rho_cnt = None

    # Atomic volume (periodic meaningful)
    atomic_vol = vol / natoms if natoms > 0 else None

    bonds = _count_bonds(s, res.is_molecule_like)

    rows = [
        ("Structure", res.name),
        ("Atoms (N)", natoms),
        ("Lattice a (Å)", a),
        ("Lattice b (Å)", b),
        ("Lattice c (Å)", c),
        ("α (deg)", alpha),
        ("β (deg)", beta),
        ("γ (deg)", gamma),
        ("Cell volume (Å³)", vol),
        ("Atomic volume (Å³/atom)", atomic_vol),
        ("Bonds (approx.)", bonds),
        ("Density (g/cm³) [unit-cell]", rho),
    ]
    if rho_cnt is not None:
        rows.append(("CNT apparent density (g/cm³) [solid-cylinder]", rho_cnt))

    # Expose key physical constants for auditability
    rows.extend(
        [
            ("Planck h (J·s)", float(sc.h)),  # EN: Planck constant; AR: ثابت بلانك
            ("Boltzmann kB (J/K)", float(sc.k)),  # EN: Boltzmann; AR: ثابت بولتزمان
            ("Avogadro NA (1/mol)", float(sc.N_A)),  # EN: Avogadro; AR: عدد أفوجادرو
            ("Atomic mass constant m_u (kg)", float(sc.atomic_mass)),  # EN/AR: وحدة الكتلة الذرية
        ]
    )

    df = pd.DataFrame(rows, columns=["Property", "Value"])
    return df


# -----------------------------
# Export helpers
# -----------------------------


def xyz_from_structure(struct: Structure, comment: str = "generated_by_streamlit") -> str:
    """
    Create an XYZ string.
    XYZ format:
      line1: number of atoms
      line2: comment
      next:  Element x y z   (Å)
    """
    coords = np.array(struct.cart_coords, dtype=float)
    species = [str(s) for s in struct.species]
    lines = [str(len(species)), str(comment)]
    for el, (x, y, z) in zip(species, coords):
        lines.append(f"{el:2s} {x: .8f} {y: .8f} {z: .8f}")
    return "\n".join(lines) + "\n"


def cif_from_structure(struct: Structure) -> str:
    """Export CIF string using pymatgen's CifWriter."""
    writer = CifWriter(struct, symprec=None) # اجعله هكذا ليتجاهل تحليل التماثل المعقد
    return writer.__str__()


# -----------------------------
# Viewer (py3Dmol -> HTML)
# -----------------------------


def _style_dict(style_name: str) -> Dict:
    if style_name == "Stick":
        return {"stick": {"radius": 0.18}}
    if style_name == "Ball & Stick":
        return {"stick": {"radius": 0.14}, "sphere": {"scale": 0.30}}
    if style_name == "Sphere":
        return {"sphere": {"scale": 0.60}}
    return {"stick": {"radius": 0.18}}


def render_3dmol_xyz(
    xyz: str,
    style_name: str,
    spin: bool,
    zoom_factor: float,
    height_px: int = 520,
) -> None:
    """
    Render xyz content using py3Dmol within Streamlit.
    Uses showmol to display; zoomTo() ensures camera is directed at atoms.
    """
    view = py3Dmol.view(width=900, height=height_px)
    view.addModel(xyz, "xyz")
    view.setStyle(_style_dict(style_name))
    # zoomTo() before display: camera centered on atoms, not empty space
    view.zoomTo()
    try:
        if abs(float(zoom_factor) - 1.0) > 1e-6:
            view.zoom(float(zoom_factor))
    except Exception:
        pass
    if spin:
        try:
            view.spin(True)
        except Exception:
            pass
    view.setBackgroundColor("#0b1020")
    showmol(view, height=500, width=700)


# -----------------------------
# Streamlit App
# -----------------------------


def main() -> None:
    st.set_page_config(
        page_title="Material Simulation Lab",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    st.markdown(
        """
        <div style="padding: 6px 2px 2px 2px;">
          <h1 style="margin-bottom: 0.2rem;">Material Simulation Lab</h1>
          <div style="opacity:0.85; font-size: 0.95rem;">
            Streamlit + py3Dmol + Pymatgen/RDKit + SciPy constants — a compact, professional materials dashboard.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### Builder")
        preset = st.selectbox("Structure type", ["Diamond", "Gold", "Silicon", "Graphene", "CNT"])

        st.markdown("### Viewer")
        style_name = st.selectbox("Render style", ["Stick", "Ball & Stick", "Sphere"], index=1)
        spin = st.toggle("Spin", value=True)
        zoom_factor = st.slider("Zoom", min_value=0.6, max_value=2.2, value=1.0, step=0.05)
        height_px = st.slider("Viewer height (px)", min_value=360, max_value=760, value=540, step=20)

        st.markdown("### Performance")
        st.caption("Keep supercells small to stay within 6GB RAM.")

    # Builder parameters (shown in main area)
    left, right = st.columns([0.44, 0.56], gap="large")

    with left:
        with st.container():
            st.markdown("### Structure Builder")

            if preset in {"Diamond", "Silicon", "Gold"}:
                default_a = {"Diamond": 3.567, "Silicon": 5.431, "Gold": 4.078}[preset]
                lattice_a = st.number_input("Lattice parameter a (Å)", min_value=1.0, max_value=20.0, value=float(default_a), step=0.001)
                graphene_vac_c = 20.0
            elif preset == "Graphene":
                lattice_a = st.number_input("Lattice parameter a (Å)", min_value=1.0, max_value=10.0, value=float(_A_GRAPHENE_ANG), step=0.001)
                graphene_vac_c = st.number_input("Vacuum c (Å)", min_value=8.0, max_value=60.0, value=20.0, step=1.0)
            else:  # CNT
                lattice_a = float(_A_GRAPHENE_ANG)
                graphene_vac_c = 20.0

            if preset != "CNT":
                st.markdown("#### Supercell (Visualization)")
                nx = st.slider("nx", 1, 4, 2)
                ny = st.slider("ny", 1, 4, 2)
                nz = st.slider("nz", 1, 4, 2)
                supercell = (int(nx), int(ny), int(nz))
                cnt_n = 10
                cnt_m = 0
                cnt_length = 40.0
            else:
                st.markdown("#### Carbon Nanotube (CNT)")
                cnt_n = st.number_input("Chirality n", min_value=1, max_value=80, value=10, step=1)
                cnt_m = st.number_input("Chirality m", min_value=0, max_value=80, value=0, step=1)
                cnt_length = st.number_input("Length (Å)", min_value=5.0, max_value=800.0, value=60.0, step=5.0)
                supercell = (1, 1, 1)

            build = st.button("Generate / Update Structure", use_container_width=True)

            st.caption(
                "Tip: For crystals, start with a 2×2×2 supercell. For CNT, larger (n,m) increases diameter and atoms."
            )

    # Build structure (cached)
    if "build_result" not in st.session_state:
        # Initial build (no button required)
        st.session_state.build_result = build_structure_cached(
            preset=preset,
            lattice_a=float(lattice_a),
            graphene_vac_c=float(graphene_vac_c),
            supercell=tuple(supercell),
            cnt_n=int(cnt_n),
            cnt_m=int(cnt_m),
            cnt_length=float(cnt_length),
        )
    elif build:
        st.session_state.build_result = build_structure_cached(
            preset=preset,
            lattice_a=float(lattice_a),
            graphene_vac_c=float(graphene_vac_c),
            supercell=tuple(supercell),
            cnt_n=int(cnt_n),
            cnt_m=int(cnt_m),
            cnt_length=float(cnt_length),
        )

    res: BuildResult = st.session_state.build_result
    xyz = xyz_from_structure(res.structure, comment=res.name)

    with right:
        with st.container():
            st.markdown("### 3D Interactive Viewer")
            st.caption("Powered by py3Dmol. Use style controls in the sidebar.")
            render_3dmol_xyz(
                xyz=xyz,
                style_name=style_name,
                spin=spin,
                zoom_factor=float(zoom_factor),
                height_px=int(height_px),
            )

    st.divider()

    tab_builder, tab_analysis, tab_export, tab_about = st.tabs(["Builder Summary", "Physical Analysis", "Export", "About / Notes"])

    with tab_builder:
        col1, col2 = st.columns([0.55, 0.45], gap="large")
        with col1:
            with st.container():
                st.markdown("### Current Structure")
                st.write(f"**Name:** {res.name}")
                st.write(f"**Atoms:** {len(res.structure)}")
                if res.metadata:
                    st.json(res.metadata, expanded=False)

        with col2:
            with st.container():
                st.markdown("### Quick Geometry")
                lat = res.structure.lattice
                df = pd.DataFrame(
                    [
                        ("a (Å)", float(lat.a)),
                        ("b (Å)", float(lat.b)),
                        ("c (Å)", float(lat.c)),
                        ("α (deg)", float(lat.alpha)),
                        ("β (deg)", float(lat.beta)),
                        ("γ (deg)", float(lat.gamma)),
                        ("Volume (Å³)", float(lat.volume)),
                    ],
                    columns=["Metric", "Value"],
                )
                st.dataframe(df, width='stretch', hide_index=True)

    with tab_analysis:
        with st.container():
            st.markdown("### Physical Analysis Engine")
            st.caption(
                "All constants shown below come from `scipy.constants` (CODATA). "
                "Bond count is approximate (RDKit-based when available, otherwise distance-cutoff)."
            )
            props = summarize_properties(res)
            props = props.copy()
            props["Value"] = props["Value"].astype(str)
            st.dataframe(props, width='stretch', hide_index=True)

    with tab_export:
        with st.container():
            st.markdown("### Scientific Export")
            st.caption("Download your generated structure as `.xyz` (universal) or `.cif` (crystallographic).")

            xyz_bytes = xyz.encode("utf-8")
            try:
                cif_bytes = cif_from_structure(res.structure).encode("utf-8")
            except Exception:
                cif_bytes = b"CIF Error"
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download XYZ",
                    data=xyz_bytes,
                    file_name=f"{res.name.replace(' ', '_')}.xyz",
                    mime="chemical/x-xyz",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "Download CIF",
                    data=cif_bytes,
                    file_name=f"{res.name.replace(' ', '_')}.cif",
                    mime="chemical/x-cif",
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("#### Preview (first lines)")
            st.code("\n".join(xyz.splitlines()[:8]), language="text")

    with tab_about:
        with st.container():
            st.markdown("### Notes")
            st.markdown(
                """
                - **Accuracy**: Physical constants (Planck, Boltzmann, Avogadro, etc.) are taken directly from `scipy.constants`.
                - **CNT density**: The app reports two density values:
                  - Unit-cell density (from the simulation box) — *not meaningful for vacuum-padded CNT boxes*
                  - CNT "apparent" density assuming a solid cylinder — useful as a scalar but not the true hollow-tube density
                - **Memory**: Keep supercells small (≤4×4×4). Large supercells scale atoms \(\propto nx \cdot ny \cdot nz\).
                """
            )


if __name__ == "__main__":
    main()

