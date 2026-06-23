# SPDX-License-Identifier: Apache-2.0
"""HTML-table structural validation. Runs before table correction so issues
flag inconsistent column counts."""

from tensorlake_docai.tables.table_correction import validate_table_structure

CLEAN_TABLE = """
<table>
  <tr><th>A</th><th>B</th><th>C</th></tr>
  <tr><td>1</td><td>2</td><td>3</td></tr>
  <tr><td>4</td><td>5</td><td>6</td></tr>
</table>
"""

INCONSISTENT_TABLE = """
<table>
  <tr><th>A</th><th>B</th><th>C</th></tr>
  <tr><td>1</td><td>2</td></tr>
</table>
"""

COLSPAN_TABLE = """
<table>
  <tr><th colspan="2">Header</th><th>C</th></tr>
  <tr><td>1</td><td>2</td><td>3</td></tr>
</table>
"""


def test_clean_table_no_issues():
    assert validate_table_structure(CLEAN_TABLE) == []


def test_short_row_is_flagged():
    issues = validate_table_structure(INCONSISTENT_TABLE)
    assert issues, "expected an issue for the row with fewer cells"
    assert any("Row 2" in i for i in issues)


def test_colspan_accounted_for():
    # colspan=2 + 1 cell == 3 cols on row 1, matches 3 cols on row 2
    assert validate_table_structure(COLSPAN_TABLE) == []


def test_empty_string_input():
    issues = validate_table_structure("")
    # Either no rows ("empty") or a parse failure — both surface a message
    assert issues
