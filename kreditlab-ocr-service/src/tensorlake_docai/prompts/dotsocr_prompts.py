# SPDX-License-Identifier: Apache-2.0
"""
Centralized DotsOCR prompts used across the `dots-ocr` modules.
"""

DOTSOCR_LAYOUT_PROMPT = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""

DOTSOCR_TABLE_PROMPT = """Convert the table in this image to HTML."""


# 1. Define the System Role (Crucial for 9b models to stay on task)
system_role = (
    "You are a specialized OCR and Data Extraction engine. "
    "Your focus is 100% accuracy in transcription and syntax. "
    "Do not explain your output. Do not chat."
)

# 2. Define user prompts for each figure type
FIGURE_CLASSIFICATION_PROMPT = (
    "Task: Classify this figure into ONE of these categories.\n\n"
    "Definitions:\n"
    "- BARCODE: Barcode, QR code, or data matrix. horizontal or vertical\n"
    "- CHART: Data visualization (graphs, plots) with axes/legends.\n"
    "- DIAGRAM: Flowcharts, relationships, or labeled schemes.\n"
    "- TABLE: Multi-row data grid with pre-filled content for reading. Contains multiple rows of data, NOT empty fields.\n"
    "  *KEY*: If the image has a REPEATING GRID structure (rows × columns like a spreadsheet), it's a TABLE even if some cells are empty.\n"
    "- FORM: Document containing *labeled* input fields (lines/boxes) meant for user input. Must show clear structure.\n"
    "  *KEY*: Forms have INDIVIDUAL labeled fields, NOT a multi-row grid. If it looks like a spreadsheet/table grid, it's a TABLE.\n"
    "- OTHER: Photos, illustrations, icons, stamps, signatures.\n"
    "  *IMPORTANT*: Use OTHER for blank images, white space, random noise, or isolated lines with no text.\n"
    "  *IMPORTANT*: Use OTHER for blurry, skewed, or unreadable images.\n\n"
    "Decision Logic:\n"
    "1. Check for Quality: If the image is blurry, heavily skewed, or text is unreadable -> Output OTHER.\n"
    "2. Check for Content: If the image is mostly empty, blank, or has no text/labels -> Output OTHER.\n"
    "3. Grid Structure Check: If it has a REPEATING row × column grid structure (like a spreadsheet) -> TABLE.\n"
    "4. Visual Data Check: If it has bars, lines, points, pie slices, or axes with plotted data -> CHART.\n"
    "5. Text Grid Check: If it's rows/columns of text WITHOUT graphical elements -> TABLE.\n"
    "6. Input Fields Check: If it has labeled INDIVIDUAL input fields (not a grid) -> FORM.\n\n"
    "Output ONLY the category name."
)


FIGURE_CHART_EXTRACTION_PROMPT = (
    f"{system_role}\n\n"
    "Task: Extract data from this chart.\n"
    "Output Format: Markdown Table\n\n"
    "Requirements:\n"
    "1. Extract the Chart Title and Axis Labels.\n"
    "2. Extract every visible data point and its value.\n"
    "3. Include Legend items.\n"
    "4. Transcribe any annotations exactly.\n"
    "5. Return ONLY the table."
)

FIGURE_DIAGRAM_EXTRACTION_PROMPT = (
    f"{system_role}\n\n"
    "Task: Transcribe the provided diagram text into structured code.\n\n"
    "Rules:\n"
    "1. Content: Preserve the text exactly in its original language. Do not translate.\n"
    "2. Format Selection:\n"
    "   - IF the image is a table or list: Output a Markdown table.\n"
    "   - IF the image is a flowchart/tree: Output Mermaid JS code.\n"
    "3. Mermaid Configuration (If Flowchart):\n"
    "   - Direction: Detect language. Arabic/Hebrew -> `graph RL`. English -> `graph LR`.\n"
    '   - Syntax: You MUST wrap ALL node text in double quotes (e.g., id["Content"]).\n'
    "4. Handling Visuals/Decorations:\n"
    "   - IF there are unconnected icons, footprints, or illustrations: IGNORE THEM.\n"
    "   - Do NOT describe them. Do NOT write a 'Note' about them.\n"
    "5. Termination:\n"
    "   - Output ONLY the code block.\n"
    "   - Stop generating immediately after the closing ``` marks."
)


FIGURE_FORM_EXTRACTION_PROMPT = (
    "Read this form and extract all visible fields.\n\n"
    "Return HTML only.\n"
    "Use one or more <table> elements.\n"
    "Each row must be exactly:\n"
    "<tr><td>field</td><td>value</td></tr>\n\n"
    "Rules:\n"
    "- Scan top to bottom, left to right.\n"
    "- Keep field labels short but clear.\n"
    "- For text or number fields, output the visible value.\n"
    "- If a field is blank, output: empty\n"
    "- For checkboxes or radio buttons, output the selected option label.\n"
    "- If none are selected, output: empty\n"
    "- For a single checkbox, output: checked or empty\n"
    "- For dates split across boxes, merge into one value.\n"
    "- For table-like regions in the form, flatten each row into field/value pairs.\n"
    "- Do not explain anything.\n"
    "- Do not include markdown fences.\n"
    "- Do not output text outside HTML.\n\n"
    "Output example:\n"
    "<table>\n"
    "<tr><td>Patient Name</td><td>John Smith</td></tr>\n"
    "<tr><td>Date of Birth</td><td>03/15/1985</td></tr>\n"
    "<tr><td>Sex</td><td>M</td></tr>\n"
    "<tr><td>Relationship</td><td>Spouse</td></tr>\n"
    "<tr><td>Signature</td><td>empty</td></tr>\n"
    "</table>"
)

FIGURE_TABLE_EXTRACTION_PROMPT = (
    f"{system_role}\n\n"
    "Task: Convert this image table into HTML.\n\n"
    "Rules:\n"
    "1. Output valid HTML tags: <table>, <thead>, <tbody>, <tr>, <td>, <th>.\n"
    "2. Content: Transcribe text exactly. Do not summarize or correct typos.\n"
    "3. Structure: Preserve exact row/column alignment.\n"
    "4. Output: Return ONLY the HTML code block."
)


# FIGURE_CAPTION_PROMPT = (
#     f"{system_role}\n\n"
#     "Task: Generate a literal visual description.\n\n"
#     "Rules:\n"
#     "1. Be robotic and minimal. No storytelling.\n"
#     "2. IF handwritten text/signature: Transcribe it exactly.\n"
#     "3. IF visual scene: List primary elements (people, objects, logos).\n"
#     "4. Format: Single line or comma-separated list."
# )
FIGURE_CAPTION_PROMPT = (
    f"{system_role}\n\n"
    "Task: Extract visible text and minimally describe the figure.\n\n"
    "Rules:\n"
    "1. IF the figure contains ONLY text (printed or handwritten, any language):\n"
    "   - Output ONLY the transcribed text.\n"
    "   - Do NOT describe background, color, layout, or style.\n\n"
    "2. IF the figure contains text AND other visual elements (e.g. logo, icon, product, person):\n"
    "   - First, transcribe all readable text exactly.\n"
    "   - Then, add ONE short phrase describing the non-text element.\n"
    "   - Keep description under 10 words.\n\n"
    "3. IF the figure contains NO readable text:\n"
    "   - Provide a very brief factual description (max 1 sentence).\n\n"
    "4. Be literal, concise, and factual. No storytelling.\n"
)
