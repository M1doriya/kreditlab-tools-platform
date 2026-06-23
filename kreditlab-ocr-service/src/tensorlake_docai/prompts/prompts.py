# SPDX-License-Identifier: Apache-2.0
# pylint: disable=line-too-long
# flake8: noqa: E501
from tensorlake_docai.pipeline.api import PageFragmentType

TABLE_EXTRACTOR_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting table data from images."""
)

TABLE_EXTRACTOR_USER_PROMPT = """[INST] Extract information from the table image as structured data. 
* Format the output as markdown. 
* Strictly return the extracted output only, and nothing else. Don't add any other text in the beggining of the output.
* If there are radio boxes or checkboxes, only include the text around the checkbox thats checked or marked. If none of the checkboxes are checked, return an empty string.
* INCLUDE ALL the information on the table.
* Don't add backticks or any other formating information like ```markdown or ```json in the output
[/INST]
"""

FIGURE_EXTRACTOR_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting figure data from images."""
)

FIGURE_EXTRACTOR_USER_PROMPT = """Extract information from the figure:

* If it is a table: Convert to structured markdown with title and caption
* If it is a graph or chart: Identify type and describe key data trends
* If it contains text: Extract all printed or handwritten content
* If it's a mathematical formula: Transcribe using LaTeX notation
* If it has checkboxes: Only include text for selected options (empty string if none)

Return only extracted information in markdown format without additional commentary.
"""


FORM_EXTRACTOR_USER_PROMPT = """[INST] Extract information from the document fragment:

* If it is a table, extract the table data into a structured markdown format, along with table title and caption. 
* If it is a graph or chart, describe it in detail.
* If it contains printed or handwritten text, extract all the text.
* If there are radio boxes or checkboxes, only include the text around the checkbox thats checked or marked. If none of the checkboxes are checked, return an empty string.
* If it is a form, extract the form data into a structured markdown format, with question and answer pairs.
Don't add ```text or ```markdown in the output.
[/INST]
"""

DOCUMENT_INDEX_SYSTEM_PROMPT = (
    """You are a helpful AI assistant specialized in extracting document index from images."""
)

DOCUMENT_INDEX_USER_PROMPT = """[INST] This is a cropped image from a document. This is a document index.Extract information from the document index:
* Extract the document index from the image.
* Return the document index in a structured markdown format. Don't include any additional markup like ```markdown or ```json.
[/INST]
"""

OCR_SYSTEM_PROMPT = """You are an OCR modelspecialized in extracting text from images."""
OCR_PROMPT = """
(<image>./</image>)
[INST]
Extract the text in the image. If this is written, printed or text, only extract the text. If it's a picture with some text on the scene, extract only the text. 
Don't describe the image. 
Don't add any other text or information at the beginning of the text.
Don't add ```text or ```markdown in the output.
ABSOLUTELY DON'T REPEAT THE TEXT OR HALLUCINATE ANYTHING. 
IF THE IMAGE IS EMPTY RETURN |notext| 
[/INST]
"""

TABLE_SUMMARY_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in summarizing table data from images efficiently and concisely."""
TABLE_SUMMARY_USER_PROMPT = """Summarize the key information from the table image in 2 to 3 lines.  

* If the table alone lacks sufficient context, refer to the full-page image for additional details.  
* Ensure the summary remains concise and does not replicate the table data.  
* Format the output in markdown.  
* Return only the extracted summary, do not include any introductory or additional text.  
"""

TABLE_SUMMARY_USER_PROMPT_NO_PAGE = """Summarize the key information from the table image in 2 to 3 lines.  

* Focus on the content within the table image itself.  
* Ensure the summary remains concise and does not replicate the table data.  
* Format the output in markdown.  
* Return only the extracted summary, do not include any introductory or additional text.  
"""

FIGURE_SUMMARY_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in summarizing figures from images efficiently and concisely."""
FIGURE_SUMMARY_USER_PROMPT = """Summarize the key information from the figure image in 2 to 3 lines.  

* If the figure alone lacks sufficient context, refer to the full-page image for additional details.  
* Ensure the summary remains concise and does not replicate the figure data.  
* Format the output in markdown.  
* Return only the extracted summary, do not include any introductory or additional text.  
"""

FIGURE_SUMMARY_USER_PROMPT_NO_PAGE = """Summarize the key information from the figure image in 2 to 3 lines.  

* Focus on the content within the figure image itself.  
* Ensure the summary remains concise and does not replicate the figure data.  
* Format the output in markdown.  
* Return only the extracted summary, do not include any introductory or additional text.  
"""

chart_schemas = {
    "pie_chart_schema": {
        "type": "pie_chart",
        "bbox": "list[number] - [ymin, xmin, ymax, xmax] bounding box of the chart",
        "chart_description": "string - detailed description of the chart",
        "title": "string - chart title",
        "data": [
            {
                "label": "string - slice label",
                "value": "number - slice value",
                "percentage": "number - slice percentage (optional, can be calculated)",  # noqa: E501
            },
        ],
        "colors": ["string - hex color codes (optional)"],
        "show_percentages": "boolean - whether to display percentages",
        "explode": ["number - separation distance for each slice (optional)"],
    },
    "bar_chart_schema": {
        "type": "bar_chart",
        "bbox": "list[number] - [ymin, xmin, ymax, xmax] bounding box of the chart",
        "chart_description": "string - detailed description of the chart",
        "title": "string - chart title",
        "orientation": "string - 'vertical' or 'horizontal'",
        "x_axis": {
            "label": "string - x-axis label",
            "categories": ["string - category labels"],
        },
        "y_axis": {
            "label": "string - y-axis label",
            "min": "number - minimum value (optional)",
            "max": "number - maximum value (optional)",
            "format": "string - 'percentage', 'currency', 'number' (optional)",
        },
        "series": [
            {
                "name": "string - series name (for grouped/stacked bars)",
                "data": ["number - data values"],
                "color": "string - hex color (optional)",
                "show_values": "boolean - show values on bars",
            },
        ],
        "bar_style": "string - 'grouped', 'stacked', 'single'",
        "grid": "boolean - whether to show grid",
    },
    "line_chart_schema": {
        "type": "line_chart",
        "bbox": "list[number] - [ymin, xmin, ymax, xmax] bounding box of the chart",
        "chart_description": "string - detailed description of the chart",
        "title": "string - chart title",
        "subtitle": "string - chart subtitle (optional)",
        "x_axis": {
            "label": "string - x-axis label",
            "values": ["number or string - x-axis values"],
            "scale": "string - 'linear', 'log', etc.",
        },
        "y_axis": {
            "label": "string - y-axis label",
            "min": "number - minimum value (optional)",
            "max": "number - maximum value (optional)",
            "scale": "string - 'linear', 'log', etc.",
        },
        "series": [
            {
                "name": "string - series name",
                "data": [
                    "number - data points (extract a dense sequence of values to represent the curve)"
                ],
                "color": "string - hex color (optional)",
                "line_style": "string - '-', '--', '-.', ':' (optional)",
                "marker": "string - 'o', 's', '^', etc. (optional)",
            },
        ],
        "legend_position": "string - 'best', 'right', 'upper left', etc. (optional)",
        "grid": "boolean - whether to show grid",
    },
    "scatter_plot_schema": {
        "type": "scatter_plot",
        "bbox": "list[number] - [ymin, xmin, ymax, xmax] bounding box of the chart",
        "chart_description": "string - detailed description of the chart",
        "title": "string - chart title",
        "subtitle": "string - chart subtitle (optional)",
        "x_axis": {
            "label": "string - x-axis label",
            "min": "number - minimum value (optional)",
            "max": "number - maximum value (optional)",
            "scale": "string - 'linear', 'log', etc.",
        },
        "y_axis": {
            "label": "string - y-axis label",
            "min": "number - minimum value (optional)",
            "max": "number - maximum value (optional)",
            "scale": "string - 'linear', 'log', etc.",
        },
        "series": [
            {
                "name": "string - series name",
                "x_data": ["number - x coordinates"],
                "y_data": ["number - y coordinates"],
                "color": "string - hex color (optional)",
                "marker": "string - 'o', 's', '^', 'D', etc. (optional)",
                "size": "number - marker size (optional)",
                "alpha": "number - transparency 0-1 (optional)",
                "edge_color": "string - marker edge color (optional)",
            },
        ],
        "legend_position": "string - 'best', 'right', 'upper left', etc. (optional)",
        "grid": "boolean - whether to show grid",
    },
    "kaplan_meier_schema": {
        "type": "kaplan_meier",
        "bbox": "list[number] - [ymin, xmin, ymax, xmax] bounding box of the chart",
        "chart_description": "string - detailed description of the chart",
        "title": "string - chart title",
        "x_axis": {
            "label": "string - x-axis label",
        },
        "y_axis": {
            "label": "string - y-axis label",
        },
        "series": [
            {
                "name": "string - series name",
                "data": [
                    "list[number] - [time, probability] pairs. Use a list of 2-element arrays (e.g. [[0, 1.0], [1.5, 0.9], ...]) for compactness. Extract a VERY high density of points to capture every step and tick mark."
                ],
            }
        ],
        "risk_table": {
            "time_points": ["number - time points"],
            "rows": [{"name": "string - series name", "values": ["number - number at risk"]}],
        },
    },
}

CHART_PROMPTS = {
    "detection": {
        "system": "You are an expert at detecting charts in images, and extracting their bounding boxes. Charts may include pie charts, bar charts, line charts, scatter plots, and Kaplan-Meier plots, nothing else.",
        "user": "Analyze this image and identify the bounding boxes of all individual charts present. Charts may include pie charts, bar charts, line charts, scatter plots, and Kaplan-Meier plots, nothing else. Return a JSON list of objects, where each object has a 'bbox' field with [ymin, xmin, ymax, xmax] coordinates (normalized 0-1000). If there are multiple charts (e.g. subplots), return a bbox for each. If there is only one chart, return a single bbox covering the chart area. For Kaplan-Meier plots, exclude the risk table from the main 'bbox', but include a separate 'risk_table_bbox' field for the risk table if present. Also, extract the 'series_names' if a legend or risk table is visible, to ensure consistency between chart and table. If no charts are present, return an empty list.",
    },
    "extraction": {
        "system": "You are a high-precision Chart Digitizer. Your mission is to recover the underlying raw dataset from chart images with maximum fidelity. You prioritize data density and accuracy over conciseness.",
        "user": f"Analyze this image and extract the chart information in JSON. The image may contain multiple charts. You must process EACH chart individually and extract ALL data points. \n\nSTRATEGY: 1. Scan the image to identify ALL charts. 2. For each chart, identify its bounding box and type. 3. Extract the data points with EXTREME density.\n\nCRITICAL INSTRUCTION: Do not summarize, downsample, or simplify. We need the raw data points for high-precision reconstruction. \n- For Kaplan-Meier and Line Charts: Perform DENSE SAMPLING. Do not rely solely on x-axis ticks. You must extract intermediate points between ticks to capture the full shape of the curve. Imagine tracing the line with a pen and recording coordinates frequently.\n- For Scatter Plots: Extract every single dot.\n- Use compact JSON formats (e.g. lists of numbers instead of objects) where possible to maximize data density within output limits.\n\nWhen multiple charts are present, do not compromise on the density of any single chart. Treat each chart as a standalone high-precision extraction task. If the output is very long, that is expected.\n\nReturn a JSON list of chart objects. Use the following schemas as a guide for the structure of each chart object:\n\nPie Chart: {chart_schemas['pie_chart_schema']}\nBar Chart: {chart_schemas['bar_chart_schema']}\nLine Chart: {chart_schemas['line_chart_schema']}\nScatter Plot: {chart_schemas['scatter_plot_schema']}\nKaplan-Meier: {chart_schemas['kaplan_meier_schema']}",
    },
    "risk_table_extraction": {
        "system": "You are an expert at extracting data from Kaplan-Meier risk tables.",
        "user": 'Analyze this image which contains a risk table from a Kaplan-Meier plot. Extract the time points and the number at risk for each series. Return a JSON object with the following structure: {"time_points": [number], "rows": [{"name": "string", "values": [number]}]}. Capture all values accurately.',
    },
}

TABLE_MERGING_PROMPTS = {
    "merged_summary": {
        "system": (
            "You are a helpful assistant that summarizes merged tables. "
            "Given a merged HTML table, provide a concise summary of its contents."
        ),
        "user": "Given the following merged HTML table:\n{merged_table}\n\n"
        "Provide a concise summary of the table's contents, highlighting key information and any notable patterns. "  # noqa E501
        'Return the output as JSON in the format: {{\n  "summary": "concise summary here"\n}}',  # noqa E501
    },
    "fast_merge": {
        "system": (
            "You are a helpful assistant that analyzes HTML tables to determine if they should be merged. "
            "The output is valid JSON."
        ),
        "user": "Given the following HTML table from the end of the first page:\n{table_end}\n\n"
        "Context between tables (e.g. page footers/headers):\n{context_between}\n\n"
        "And the following HTML table from the start of the second page:\n{table_start}\n\n"
        "Step 1: Determine if the second table is a continuation of the first table.\n"
        "Step 2: If 'YES', provide an explanation as to why and the number of rows to skip from the start of the second table (0 if none). This refers to header rows that could be removed to improve merging.\n"
        "If 'NO', do not merge.\n\n"
        "Answer in the following JSON format:\n"
        "{{\n"
        "  'continuation': 'YES' or 'NO',\n"
        "  'explanation': 'brief explanation',\n"
        "  'skip_rows': 'number of rows to skip.'\n"
        "}}",
    },
    "fast_same_page_merge": {
        "system": (
            "You are a helpful assistant that analyzes HTML tables on the same page to determine if they should be merged. "
            "The output is valid JSON."
        ),
        "user": "Table 1:\n{table1}\n\n"
        "Context between tables (e.g. page footers/headers):\n{context_between}\n\n"
        "Table 2:\n{table2}\n\n"
        "Step 1: Determine if Table 2 is a continuation of Table 1 (e.g. split by layout).\n"
        "Step 2: If 'YES', determine how many header rows at the beginning of Table 2 repeat headers from Table 1 and should be skipped during merging.\n"
        "Step 3: If 'YES' and the column structure of Table 2 appears misaligned with Table 1, provide a corrected HTML for the *entire* Table 2 to ensure it aligns properly. For the alignment, pay attention to the type of content of the columns both syntactic and semantically, so they two tables are best aligned. If no correction is needed, this field can be null.\n"
        "If 'NO', do not merge.\n\n"
        "Answer in the following JSON format:\n"
        "{{\n"
        "  'should_merge': 'YES' or 'NO',\n"
        "  'explanation': 'brief explanation',\n"
        "  'skip_rows': <number of rows to skip from the start of Table 2 (0 if none)>,\n"
        "  'corrected_html': <corrected HTML for Table 2 if needed, otherwise null>\n"
        "}}",
    },
    "align_tables": {
        "system": "You are an expert in semantically aligning HTML table columns based on their visual representation for merging tables.",
        "user": "The following two tables are from consecutive pages of a document and are part of the same semantic table. "
        "However, their column structures in HTML do not align semantically.\n"
        "Table 1 (End of previous page):\n"
        " {table1_html} \n\n"
        "Table 2 (Start of current page):\n"
        " {table2_html} \n\n"
        "You are provided with images of both tables.\n"
        "Your task is to generate a CORRECTED HTML for Table 2 so that its columns align perfectly semantically with Table 1.\n"
        "- Ensure Table 2 has the same number of columns as Table 1.\n"
        "- Adjust column spans or add empty cells or remove cells if necessary to match the semantic alignment.\n"
        "- The content of Table 2 must be preserved, just restructured cells to align with Table 1.\n"
        "- Do not repeat content not present in the Table 2 image.\n"
        "- Return the corrected HTML and an explanation as JSON in the format: {{ \"corrected_html': '<html>...</html>\", \"explanation\": '<explanation>'}}\n",
    },
}

TABLE_CORRECTION_PROMPTS = {
    "system": (
        "You are an expert in HTML table correction. "
        "You will receive an image of a table, an extracted HTML version, and an error analysis report. "
        "Your task is to fix the HTML table to match the image and resolve the reported errors."
    ),
    "user": "You are an expert in HTML table correction. "
    "You will receive an image of a table and an HTML version. "
    "Your task is to fix the HTML table to match the image and resolve the reported errors and any other error. "
    "Analyze the structure of the table in the image, look at table lines and gaps, and use it as reference to correct the HTML so the rows and columns are reflected accurately. "
    "Use the table headers, cell contents, and overall layout as clues to fix merged or split cells, misaligned data, and any other issues. "
    "The HTML does not need to contain any style or formatting, focus solely on the structure and content accuracy. "
    "In the cell content, there is no need to include characters when added to fill empty spaces, e.g. with dots or hyphens.\n\n"
    "HTML Table:\n{html_input}\n\n"
    "Error Analysis Report:\n{error_report}\n\n"
    'The output is in JSON with a "corrected_html" field and an "explanation" field.',
}

FORM_PROMPTS = {
    "detection": {
        "system": "You are an expert at detecting forms in images.",
        "user": "Analyze this image and determine if it is a form. "
        "A form is a document with labeled fields for user input (e.g. text boxes, checkboxes, radio buttons). "
        "Return a JSON object with a single boolean field 'is_form'.",
    },
    "extraction": {
        "system": "You are an expert at extracting data from forms.",
        "user": "Extract the information from this form. Represent key-value pairs clearly. "
        "For checkboxes/radio buttons, indicate their state (e.g., [x] for checked, [ ] for unchecked). "
        "Consider the context of the form fields and radio buttons when processing the form image. "
        "Preserve the structure and order of the form as much as possible.",
    },
}


def get_element_summary_prompt(element_types: list[PageFragmentType], has_page_image: bool) -> str:
    """
    Get the appropriate element summary prompt based on element types and page image availability.

    Args:
        element_types: List of element types being processed
        has_page_image: Whether a full page image is available

    Returns:
        str: The appropriate user prompt for element summarization
    """
    if PageFragmentType.TABLE in element_types:
        return TABLE_SUMMARY_USER_PROMPT if has_page_image else TABLE_SUMMARY_USER_PROMPT_NO_PAGE
    elif PageFragmentType.FIGURE in element_types:
        return FIGURE_SUMMARY_USER_PROMPT if has_page_image else FIGURE_SUMMARY_USER_PROMPT_NO_PAGE
    else:
        # Fallback for other types - use table prompt as default
        return TABLE_SUMMARY_USER_PROMPT if has_page_image else TABLE_SUMMARY_USER_PROMPT_NO_PAGE


QWEN_SIGNATURE_DETECTION_SYSTEM_PROMPT = (
    """You are an assistant that detects handwritten signatures in images."""
)
QWEN_SIGNATURE_DETECTION_PROMPT = """
Spotting all signed signatures in the image with line-level,
return results in this exact JSON format:
{
    "has_signature": true/false,
    "bbox_2d": [
        [x1, y1, x2, y2],
        [x1, y1, x2, y2]
    ]
}
Where:
- "has_signature" is true if any signatures found, false otherwise
- "bbox_2d" contains the coordinates [top-left x, top-left y, bottom-right x, bottom-right y] for each signature
- If no signatures, return empty list for bbox_2d
"""

# Update form to html prompt to handle itypical forms that only have key or value
QWEN_FORM_TO_HTML_SYSTEM_PROMPT = """You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in QwenVL Document Parser HTML format using specified tags while maintaining user privacy and data integrity."""
QWEN_FORM_TO_HTML_PROMPT = """Analyze the provided image and Extract ONLY the visible text and convert to HTML. Return ONLY HTML code. Follow these strict rules:

1. For form regions with key-value pairs:
   - Use definition lists: `<dl>`, `<dt>` for keys, and `<dd>` for values
   - For standalone values without keys, use `<dd class="standalone">`
   - For keys without values, use `<dt class="no-value">`
   - Preserve visual grouping with `<div class="form-section">` when appropriate

2. For tabular data, use <table>, <tr>, <th>, <td> tags.

- Extract TEXT ONLY - no placeholders for visual elements
- Only include text actually visible in the image
- Do NOT include any comments, explanations, or non-HTML syntax
"""


QWEN_TABLE_TO_MARKDOWN_SYSTEM_PROMPT = "You are an AI assistant specialized in recognizing and extracting text from images. Your mission is to analyze the image and generate result in markdown and only use the text in the image."
QWEN_TABLE_TO_MARKDOWN_PROMPT = "Convert the document fragment to markdown. Do not include the text not on the image, do not include other outputs."

# HTML versions of table prompts
QWEN_TABLE_TO_HTML_SYSTEM_PROMPT = "You are an AI assistant specialized in recognizing and extracting text from images. Your mission is to analyze the image and generate result in HTML and only use the text in the image."
QWEN_TABLE_TO_HTML_PROMPT = "Convert the document fragment to HTML. Do not include the text not on the image, do not include other outputs."

STRUCTURED_EXTRACTION_SYSTEM_PROMPT = """You are an AI specialized in recognizing and extracting text from images. Your mission is to analyze the image document and generate the result in the given JSON schema. Return null if you can't find the data in the image."""
STRUCTURED_EXTRACTION_PROMPT = (
    """Extract the data from the image and return it in the given JSON schema."""
)


def get_prompt_messages(cls: PageFragmentType) -> list[str]:
    if cls in [PageFragmentType.TABLE]:
        return [TABLE_EXTRACTOR_SYSTEM_PROMPT, TABLE_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.FIGURE, PageFragmentType.FORMULA]:
        return [FIGURE_EXTRACTOR_SYSTEM_PROMPT, FIGURE_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.FORM, PageFragmentType.KEY_VALUE_REGION]:
        return [TABLE_EXTRACTOR_SYSTEM_PROMPT, FORM_EXTRACTOR_USER_PROMPT]
    elif cls in [PageFragmentType.DOCUMENT_INDEX]:
        return [DOCUMENT_INDEX_SYSTEM_PROMPT, DOCUMENT_INDEX_USER_PROMPT]
    elif cls in [
        PageFragmentType.TITLE,
        PageFragmentType.SECTION_HEADER,
        PageFragmentType.TEXT,
        PageFragmentType.LIST_ITEM,
        PageFragmentType.TABLE_CAPTION,
        PageFragmentType.FIGURE_CAPTION,
        PageFragmentType.FORMULA_CAPTION,
        PageFragmentType.PAGE_FOOTER,
        PageFragmentType.PAGE_HEADER,
        PageFragmentType.PAGE_NUMBER,
    ]:
        return [OCR_SYSTEM_PROMPT, OCR_PROMPT]
    else:
        raise ValueError(f"Invalid class name: {cls}")


def _get_prompt_messages_from_dict(task: str, prompt_dict: dict, task_type: str) -> list[str]:
    if task in prompt_dict:
        prompt_data = prompt_dict[task]
        return [prompt_data["system"], prompt_data["user"]]
    else:
        raise ValueError(f"Invalid {task_type} task: {task}")


def get_chart_prompt_messages(task: str) -> list[str]:
    return _get_prompt_messages_from_dict(task, CHART_PROMPTS, "chart")


def get_table_merging_prompt_messages(task: str) -> list[str]:
    return _get_prompt_messages_from_dict(task, TABLE_MERGING_PROMPTS, "table merging")


def get_table_correction_prompt_messages(additional_prompt: str | None = None) -> list[str]:
    return TABLE_CORRECTION_PROMPTS["system"], TABLE_CORRECTION_PROMPTS["user"]


def get_form_prompt_messages(task: str) -> list[str]:
    return _get_prompt_messages_from_dict(task, FORM_PROMPTS, "form")
