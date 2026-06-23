# SPDX-License-Identifier: Apache-2.0
"""
Azure Cell Bbox Extractor
Extracts cell-level bounding boxes from Azure Document Intelligence table responses
"""

from typing import List, Dict, Optional
from tensorlake_docai.models.layout_objects import TextBoundingBox


class AzureCellBboxExtractor:
    """Extracts cell-level bounding boxes from Azure Document Intelligence table responses"""

    def extract_table_cells_with_bboxes(
        self,
        table,
        page_width_pixels: int,
        page_height_pixels: int,
        page_width_inches: float,
        page_height_inches: float,
        page_number: int,
        reading_order: int,
    ) -> List[TextBoundingBox]:
        """
        Extract cell-level bounding boxes from Azure table object.

        Args:
            table: Azure table object with cells
            page_width_pixels: Page width in pixels
            page_height_pixels: Page height in pixels
            page_width_inches: Page width in inches from Azure
            page_height_inches: Page height in inches from Azure
            page_number: Page number for ref_id
            reading_order: Reading order for ref_id

        Returns:
            List of TextBoundingBox objects with cell data and ref_ids
        """
        if not hasattr(table, "cells") or not table.cells:
            print("Table has no cells or empty cells")
            return []

        # Calculate scaling factors (Azure uses inches, we need pixels)
        scale_x = page_width_pixels / page_width_inches
        scale_y = page_height_pixels / page_height_inches

        cells_with_bboxes = []

        # Prefer stable row/column ordering if available
        ordered_cells = []
        for idx, cell in enumerate(getattr(table, "cells", []) or []):
            row_idx = getattr(cell, "rowIndex", getattr(cell, "row_index", None))
            col_idx = getattr(cell, "columnIndex", getattr(cell, "column_index", None))
            ordered_cells.append(
                (
                    row_idx if row_idx is not None else 10**9,
                    col_idx if col_idx is not None else 10**9,
                    idx,
                    cell,
                )
            )
        # Sort by row, then column, fallback to original index if not present
        ordered_cells.sort(key=lambda x: (x[0], x[1], x[2]))

        for seq_idx, (row_idx, col_idx, ___, cell) in enumerate(ordered_cells):
            try:
                # Extract cell text
                cell_text = getattr(cell, "content", "")
                if not cell_text:
                    continue

                # Get bounding box from cell
                bbox = self._get_cell_bbox(cell, scale_x, scale_y)

                if bbox:
                    # Create ref_id: page.reading_order.cell_index
                    ref_id = f"{page_number}.{reading_order}.{seq_idx}"

                    cells_with_bboxes.append(
                        TextBoundingBox(
                            bbox=(bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]),
                            text=cell_text.strip(),
                            ref_id=ref_id,
                            row_index=row_idx if row_idx != 10**9 else None,
                            column_index=col_idx if col_idx != 10**9 else None,
                        )
                    )

            except Exception as e:
                print(f"Error extracting cell: {e}")
                continue

        return cells_with_bboxes

    def _get_cell_bbox(self, cell, scale_x: float, scale_y: float) -> Optional[Dict]:
        """
        Extract bounding box from a cell object.

        Args:
            cell: Azure cell object
            scale_x: Scale factor for x coordinates (pixels/inch)
            scale_y: Scale factor for y coordinates (pixels/inch)

        Returns:
            Dict with x1, y1, x2, y2 in pixels, or None if no bbox found
        """
        # Try camelCase first (Azure's typical format)
        if hasattr(cell, "boundingRegions") and cell.boundingRegions:
            region = cell.boundingRegions[0]
            if hasattr(region, "polygon") and region.polygon:
                return self._polygon_to_bbox(region.polygon, scale_x, scale_y)

        # Try snake_case (in case Azure SDK converts it)
        elif hasattr(cell, "bounding_regions") and cell.bounding_regions:
            region = cell.bounding_regions[0]
            if hasattr(region, "polygon") and region.polygon:
                return self._polygon_to_bbox(region.polygon, scale_x, scale_y)

        # Try direct polygon attribute
        elif hasattr(cell, "polygon") and cell.polygon:
            return self._polygon_to_bbox(cell.polygon, scale_x, scale_y)

        return None

    def _polygon_to_bbox(
        self, polygon: List[float], scale_x: float, scale_y: float
    ) -> Optional[Dict]:
        """
        Convert polygon points to bounding box.

        Args:
            polygon: List of points [x1, y1, x2, y2, x3, y3, x4, y4] in inches
            scale_x: Scale factor for x coordinates (pixels/inch)
            scale_y: Scale factor for y coordinates (pixels/inch)

        Returns:
            Dict with x1, y1, x2, y2 in pixels
        """
        if not polygon or len(polygon) < 8:
            return None

        # Extract x and y coordinates and convert from inches to pixels
        x_coords = [polygon[i] * scale_x for i in range(0, len(polygon), 2)]
        y_coords = [polygon[i] * scale_y for i in range(1, len(polygon), 2)]

        return {
            "x1": int(min(x_coords)),
            "y1": int(min(y_coords)),
            "x2": int(max(x_coords)),
            "y2": int(max(y_coords)),
        }
