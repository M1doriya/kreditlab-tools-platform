# SPDX-License-Identifier: Apache-2.0
import zipfile
import re
from collections import defaultdict

# Keep running counters for each numId
list_counters = defaultdict(lambda: defaultdict(int))  # {numId: {ilvl: count}}

WORD_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

# Alternative OOXML namespace (used by some documents)
WORD_NS_ALT = {
    "w": "http://purl.oclc.org/ooxml/wordprocessingml/main",
    "r": "http://purl.oclc.org/ooxml/officeDocument/relationships",
}


# formatting helper
def format_counter(val, fmt):
    if fmt == "decimal":
        return str(val)
    elif fmt == "lowerLetter":
        return chr(ord("a") + val - 1)
    elif fmt == "upperLetter":
        return chr(ord("A") + val - 1)
    elif fmt == "lowerRoman":
        return to_roman(val).lower()
    elif fmt == "upperRoman":
        return to_roman(val).upper()
    elif fmt == "bullet":
        return "•"
    else:
        return str(val)


def to_roman(num):
    vals = [
        (1000, "M"),
        (900, "CM"),
        (500, "D"),
        (400, "CD"),
        (100, "C"),
        (90, "XC"),
        (50, "L"),
        (40, "XL"),
        (10, "X"),
        (9, "IX"),
        (5, "V"),
        (4, "IV"),
        (1, "I"),
    ]
    res = ""
    for v, sym in vals:
        while num >= v:
            res += sym
            num -= v
    return res


def extract_text_with_changes_and_comments(docx_path):
    from lxml import etree

    try:
        # Reset list counters for each document
        list_counters.clear()

        # Open .docx as a zip
        with zipfile.ZipFile(docx_path) as docx:
            document_xml = docx.read("word/document.xml")
            filelist = docx.namelist()

            print("############what is in the filelist", filelist)

            styles_xml = docx.read("word/styles.xml") if "word/styles.xml" in filelist else None
            numbering_xml = (
                docx.read("word/numbering.xml") if "word/numbering.xml" in filelist else None
            )
            comments_xml = (
                docx.read("word/comments.xml") if "word/comments.xml" in filelist else None
            )
            headers_xml = [docx.read(f) for f in filelist if re.match(r"word/header\d*\.xml", f)]
            footers_xml = [docx.read(f) for f in filelist if re.match(r"word/footer\d*\.xml", f)]
            footnotes_xml = (
                docx.read("word/footnotes.xml") if "word/footnotes.xml" in filelist else None
            )
            endnotes_xml = (
                docx.read("word/endnotes.xml") if "word/endnotes.xml" in filelist else None
            )

        doc_tree = etree.fromstring(document_xml)

        # Detect which namespace variant is used in this document
        root_ns = doc_tree.tag.split("}")[0].strip("{") if "}" in doc_tree.tag else None
        if root_ns and "purl.oclc.org/ooxml" in root_ns:
            ns = WORD_NS_ALT
        else:
            ns = WORD_NS

        # --- Styles map ---
        style_map = {}
        if styles_xml:
            s_tree = etree.fromstring(styles_xml)
            for st in s_tree.xpath("//w:style", namespaces=ns):
                style_id = st.attrib.get(f"{{{ns['w']}}}styleId")
                name_el = st.find("w:name", namespaces=ns)
                if name_el is not None:
                    val = name_el.attrib.get(f"{{{ns['w']}}}val", "")
                    if re.match(r"Heading\s*(\d+)", val, re.IGNORECASE):
                        style_map[style_id] = ("heading", int(re.search(r"\d+", val).group(0)))
                    elif val.lower() == "title":
                        style_map[style_id] = ("title", None)
                    elif val.lower() == "subtitle":
                        style_map[style_id] = ("subtitle", None)
                    elif "quote" in val.lower():
                        style_map[style_id] = ("quote", None)
                    elif val.lower() == "caption":
                        style_map[style_id] = ("caption", None)

        # --- Numbering (bullets/lists) map ---
        numId_to_list = {}
        if numbering_xml:
            num_tree = etree.fromstring(numbering_xml)
            # Abstract numbering (style definitions)
            abs_map = {}
            for an in num_tree.xpath("//w:abstractNum", namespaces=ns):
                abs_id = an.attrib.get(f"{{{ns['w']}}}abstractNumId")
                # grab first level symbol/text
                lvl_elems = an.xpath("./w:lvl", namespaces=ns)
                abs_map[abs_id] = []
                for lvl in lvl_elems:
                    fmt = lvl.find("w:numFmt", namespaces=ns)
                    txt = lvl.find("w:lvlText", namespaces=ns)
                    abs_map[abs_id].append(
                        {
                            "text": (
                                txt.attrib.get(f"{{{ns['w']}}}val") if txt is not None else "%1"
                            ),
                            "fmt": (
                                fmt.attrib.get(f"{{{ns['w']}}}val")
                                if fmt is not None
                                else "decimal"
                            ),
                        }
                    )
            # Concrete numId mapping to abstract
            for num in num_tree.xpath("//w:num", namespaces=ns):
                num_id = num.attrib.get(f"{{{ns['w']}}}numId")
                abs_id_el = num.find("w:abstractNumId", namespaces=ns)
                if abs_id_el is not None:
                    abs_id = abs_id_el.attrib.get(f"{{{ns['w']}}}val")
                    numId_to_list[num_id] = abs_map.get(abs_id, [])

        # --- Comments map ---
        comments_map = {}
        if comments_xml:
            comments_tree = etree.fromstring(comments_xml)
            for c in comments_tree.xpath("//w:comment", namespaces=ns):
                cid = c.attrib.get(f"{{{ns['w']}}}id")
                comments_map[cid] = "".join(c.itertext()).strip()

        # --- Footnotes / Endnotes maps ---
        footnote_map, endnote_map = {}, {}
        if footnotes_xml:
            f_tree = etree.fromstring(footnotes_xml)
            for fn in f_tree.xpath("//w:footnote", namespaces=ns):
                fid = fn.attrib.get(f"{{{ns['w']}}}id")
                footnote_map[fid] = "".join(fn.itertext()).strip()
        if endnotes_xml:
            e_tree = etree.fromstring(endnotes_xml)
            for en in e_tree.xpath("//w:endnote", namespaces=ns):
                eid = en.attrib.get(f"{{{ns['w']}}}id")
                endnote_map[eid] = "".join(en.itertext()).strip()

        # --- Paragraph extractor ---
        def extract_paragraph(p_elem):
            out = []
            paragraph_style = None
            heading_level = None

            # Detect numbering and bullets
            pPr = p_elem.find("w:pPr", namespaces=ns)
            if pPr is not None:
                # Check heading style
                pStyle = pPr.find("w:pStyle", namespaces=ns)
                if pStyle is not None:
                    sid = pStyle.attrib.get(f"{{{ns['w']}}}val")
                    if sid in style_map:
                        kind, lvl = style_map[sid]
                        if kind == "heading":
                            heading_level = lvl
                        else:
                            paragraph_style = kind
                numPr = pPr.find("w:numPr", namespaces=ns)
                if numPr is not None:
                    numId_el = numPr.find("w:numId", namespaces=ns)
                    lvl_el = numPr.find("w:ilvl", namespaces=ns)
                    if numId_el is not None:
                        numId = numId_el.attrib.get(f"{{{ns['w']}}}val")
                        lvl = (
                            int(lvl_el.attrib.get(f"{{{ns['w']}}}val", "0"))
                            if lvl_el is not None
                            else 0
                        )

                        # get format string
                        if numId in numId_to_list and lvl < len(numId_to_list[numId]):
                            lvl_info = numId_to_list[numId][lvl]
                            fmt_pattern = lvl_info["text"]  # e.g. "%1)"

                            list_counters[numId][lvl] += 1
                            for deeper in range(lvl + 1, 9):
                                list_counters[numId][deeper] = 0

                            marker = fmt_pattern
                            for i in range(1, lvl + 2):
                                val = list_counters[numId][i - 1]
                                fmt_used = (
                                    numId_to_list[numId][i - 1]["fmt"]
                                    if i - 1 < len(numId_to_list[numId])
                                    else "decimal"
                                )
                                marker = marker.replace(f"%{i}", format_counter(val, fmt_used))

                            indent = "&nbsp;" * (lvl * 4)
                            out.append(f"{indent}{marker} ")

            def walk(node):
                if not hasattr(node, "tag"):
                    return
                tag = etree.QName(node.tag).localname

                if tag == "ins" or tag == "moveTo":
                    txt = "".join(node.itertext()).strip()
                    if txt:
                        out.append(f"<ins>{txt}</ins>")
                    return

                if tag == "del" or tag == "moveFrom":
                    txt = "".join(node.itertext()).strip()
                    if txt:
                        out.append(f"<del>{txt}</del>")
                    return

                if tag == "commentRangeStart":
                    cid = node.attrib.get(f"{{{ns['w']}}}id")
                    if cid in comments_map:
                        out.append(f'<span class="comment" data-note="{comments_map[cid]}">')

                elif tag == "commentRangeEnd":
                    out.append("</span>")

                elif tag == "commentReference":
                    cid = node.attrib.get(f"{{{ns['w']}}}id")
                    if cid in comments_map:
                        out.append(f"<!-- Comment: {comments_map[cid]} -->")

                elif tag == "hyperlink":
                    href = node.attrib.get(f"{{{ns['r']}}}id", "#")
                    text = "".join(node.itertext()).strip()
                    if text:
                        out.append(f'<a href="{href}">{text}</a>')
                    return

                elif tag == "footnoteReference":
                    fid = node.attrib.get(f"{{{ns['w']}}}id")
                    if fid in footnote_map:
                        out.append(
                            f'<sup class="footnote" data-note="{footnote_map[fid]}">[{fid}]</sup>'
                        )

                elif tag == "endnoteReference":
                    eid = node.attrib.get(f"{{{ns['w']}}}id")
                    if eid in endnote_map:
                        out.append(
                            f'<sup class="endnote" data-note="{endnote_map[eid]}">[{eid}]</sup>'
                        )

                elif tag == "r":
                    # Check for revision properties first
                    rpr = node.find("w:rPr", namespaces=ns)
                    if rpr is not None:
                        if rpr.find("w:del", namespaces=ns) is not None:
                            txt = "".join(node.itertext()).strip()
                            if txt:
                                out.append(f"<del>{txt}</del>")
                            return
                        elif rpr.find("w:ins", namespaces=ns) is not None:
                            txt = "".join(node.itertext()).strip()
                            if txt:
                                out.append(f"<ins>{txt}</ins>")
                            return

                    # Extract text, tabs, and breaks properly, then continue walking
                    for child in node:
                        child_tag = (
                            etree.QName(child.tag).localname if hasattr(child, "tag") else None
                        )
                        if child_tag == "t":
                            out.append(child.text or "")
                        elif child_tag == "tab":
                            out.append("\t")
                        elif child_tag == "br":
                            out.append("<br/>")
                        elif child_tag == "cr":
                            out.append("\n")
                        else:
                            # Continue walking for other elements like commentReference
                            walk(child)
                    return

                # Walk children (except when short-circuited)
                for child in node:
                    walk(child)

            walk(p_elem)
            final_text = "".join(out).strip()
            if not final_text:
                return ""
            if heading_level:
                return f"<h{heading_level}>{final_text}</h{heading_level}>"
            elif paragraph_style:
                return f"<{paragraph_style}>{final_text}</{paragraph_style}>"
            else:
                return f"{final_text}"

        output_parts = []

        # --- Headers ---
        for h_xml in headers_xml:
            try:
                h_tree = etree.fromstring(h_xml)
                for p in h_tree.xpath("//w:p", namespaces=ns):
                    text = extract_paragraph(p)
                    if text:
                        output_parts.append(f"<header><p>{text}</p></header>")
            except Exception as e:
                print(f"Warning: Failed to parse header: {e}")
                continue

        # --- Body (paragraphs + tables + textboxes) ---
        for elem in doc_tree.xpath("//w:body/*", namespaces=ns):
            tag = etree.QName(elem.tag).localname

            if tag == "p":
                text = extract_paragraph(elem)
                if text:
                    output_parts.append(f"<p>{text}</p>")

            elif tag == "tbl":
                table_html = ["<table border='1'>"]
                rows = elem.xpath("./w:tr", namespaces=ns)
                for r_idx, row in enumerate(rows):
                    table_html.append("<tr>")
                    cells = row.xpath("./w:tc", namespaces=ns)
                    logical_col = 0
                    for c_idx, cell in enumerate(cells):
                        # Detect column span (w:gridSpan)
                        colspan = 1
                        rowspan = 1
                        tcPr = cell.find("w:tcPr", namespaces=ns)
                        if tcPr is not None:
                            grid_span_el = tcPr.find("w:gridSpan", namespaces=ns)
                            if grid_span_el is not None:
                                colspan_val = grid_span_el.attrib.get(f"{{{ns['w']}}}val", "1")
                                try:
                                    colspan = int(colspan_val) if colspan_val else 1
                                except ValueError:
                                    colspan = 1

                            vmerge_el = tcPr.find("w:vMerge", namespaces=ns)
                            vmerge_val = (
                                vmerge_el.attrib.get(f"{{{ns['w']}}}val", "")
                                if vmerge_el is not None
                                else None
                            )

                            # Skip continuation cells for vertical merges
                            if vmerge_el is not None and (
                                vmerge_val is None or vmerge_val == "" or vmerge_val == "continue"
                            ):
                                # Continuation of a vertical merge; do not render a new <td>
                                logical_col += colspan
                                continue

                            # If this is the start of a vertical merge, compute rowspan
                            if vmerge_el is not None and vmerge_val != "continue":
                                span = 1
                                # Count how many subsequent rows continue the merge at the same logical column
                                for rr in range(r_idx + 1, len(rows)):
                                    next_cells = rows[rr].xpath("./w:tc", namespaces=ns)
                                    # Find cell at same logical column in next row
                                    next_logical_col = 0
                                    found_cell = None
                                    for nc in next_cells:
                                        if next_logical_col == logical_col:
                                            found_cell = nc
                                            break
                                        nc_tcPr = nc.find("w:tcPr", namespaces=ns)
                                        nc_colspan = 1
                                        if nc_tcPr is not None:
                                            nc_grid_span_el = nc_tcPr.find(
                                                "w:gridSpan", namespaces=ns
                                            )
                                            if nc_grid_span_el is not None:
                                                try:
                                                    nc_colspan = int(
                                                        nc_grid_span_el.attrib.get(
                                                            f"{{{ns['w']}}}val", "1"
                                                        )
                                                        or 1
                                                    )
                                                except ValueError:
                                                    nc_colspan = 1
                                        next_logical_col += nc_colspan

                                    if found_cell is None:
                                        break
                                    next_tcPr = found_cell.find("w:tcPr", namespaces=ns)
                                    if next_tcPr is None:
                                        break
                                    next_vmerge_el = next_tcPr.find("w:vMerge", namespaces=ns)
                                    if next_vmerge_el is None:
                                        break
                                    next_vmerge_val = next_vmerge_el.attrib.get(
                                        f"{{{ns['w']}}}val", ""
                                    )
                                    if (
                                        next_vmerge_val is None
                                        or next_vmerge_val == ""
                                        or next_vmerge_val == "continue"
                                    ):
                                        span += 1
                                    else:
                                        break
                                rowspan = span

                        # Gather cell inner HTML
                        cell_parts = []
                        for p in cell.xpath(".//w:p", namespaces=ns):
                            text = extract_paragraph(p)
                            if text:
                                cell_parts.append(f"<p>{text}</p>")
                        cell_html = "".join(cell_parts) if cell_parts else "&nbsp;"

                        # Build <td> with optional colspan/rowspan
                        attrs = []
                        if colspan and colspan > 1:
                            attrs.append(f"colspan='{colspan}'")
                        if rowspan and rowspan > 1:
                            attrs.append(f"rowspan='{rowspan}'")
                        attr_str = (" " + " ".join(attrs)) if attrs else ""
                        table_html.append(f"<td{attr_str}>{cell_html}</td>")
                        logical_col += colspan
                    table_html.append("</tr>")
                table_html.append("</table>")
                output_parts.append("".join(table_html))

        # --- Extract text from textboxes and shapes ---
        for textbox in doc_tree.xpath("//w:txbxContent", namespaces=ns):
            for p in textbox.xpath(".//w:p", namespaces=ns):
                text = extract_paragraph(p)
                if text:
                    output_parts.append(f"<div class='textbox'><p>{text}</p></div>")

        # --- Footers ---
        for f_xml in footers_xml:
            try:
                f_tree = etree.fromstring(f_xml)
                for p in f_tree.xpath("//w:p", namespaces=ns):
                    text = extract_paragraph(p)
                    if text:
                        output_parts.append(f"<footer><p>{text}</p></footer>")
            except Exception as e:
                print(f"Warning: Failed to parse footer: {e}")
                continue

        return "\n".join(output_parts)

    except Exception as e:
        return f"Error parsing DOCX file: {str(e)}"


if __name__ == "__main__":
    path = "test parsing docx.docx"
    html_result = extract_text_with_changes_and_comments(path)
    print(html_result)
