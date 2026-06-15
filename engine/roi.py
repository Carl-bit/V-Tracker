"""ROI de cancha (camara fija): poligono de la zona de JUEGO para filtrar el balon.

El 2o balon (banda/calentamiento) y los botes fuera de juego viven AFUERA del
poligono. Filtrar las detecciones de balon por ROI mata el balon de banda en la
fuente: mejor que el filtro de estatico de track.py (que solo agarra balones
perfectamente quietos; uno que manipulan/botan se le escapa). Camara fija => un
solo poligono sirve para todo el video.

Dibujar el poligono GENEROSO (ver scripts/pick_roi.py): cancha + margen + el aire
por encima de la red (el balon sube), EXCLUYENDO la banda lateral donde descansa
el balon extra. Un balon jugado que cae apenas fuera de las lineas debe quedar
DENTRO del ROI; solo la banda/tribuna queda afuera.

Standalone: sin FastAPI, sin ARQ (regla estructura). Coords normalizadas 0-1.
"""

import json
from pathlib import Path


class CourtROI:
    """Poligono de zona de juego en coords normalizadas; test punto-dentro."""

    def __init__(self, polygon_norm: list) -> None:
        self.polygon_norm = [(float(x), float(y)) for x, y in polygon_norm]
        if len(self.polygon_norm) < 3:
            raise ValueError(f"ROI necesita >=3 vertices, vino {len(self.polygon_norm)}")

    @classmethod
    def load(cls, path: str | Path) -> "CourtROI":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["polygon"])

    def contains_norm(self, xn: float, yn: float) -> bool:
        """Punto (xn,yn) normalizado dentro del poligono (ray casting)."""
        poly = self.polygon_norm
        inside = False
        n = len(poly)
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            denom = (yj - yi) or 1e-12
            if ((yi > yn) != (yj > yn)) and (xn < (xj - xi) * (yn - yi) / denom + xi):
                inside = not inside
            j = i
        return inside
