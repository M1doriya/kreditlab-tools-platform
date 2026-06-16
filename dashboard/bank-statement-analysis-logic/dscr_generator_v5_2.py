"""
DSCR Calculator v5.3 - JSON to HTML Generator
==============================================
Streamlit app that generates DSCR calculators from JSON data.
Loads HTML template from external file for easy maintenance.

Author: Kredit Lab
Version: 5.3 (Option C - Normalized) + 3 Audited Years + Dual Theme
Updated: April 2026

Files required:
- dscr_generator_v5_2.py (this file)
- dscr_template_v5_2.html (HTML template)

Supports JSON formats:
- v5.1: Legacy format with facilities array
- v5.2: fy1/fy2/mgmt structure
- v5.3: fy1/fy2/fy3 structure (3 audited years)
"""

import streamlit as st
import json
import re
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================
TEMPLATE_FILE = "dscr_template_v5_2.html"
VERSION = "5.3"

# =============================================================================
# PAGE CONFIG & CUSTOM CSS
# =============================================================================
st.set_page_config(
    page_title="DSCR Generator v5.3",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 28px; }
    .main-header p { margin: 5px 0 0; opacity: 0.8; }
    .preview-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .success-box {
        background: #dcfce7;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .warning-box {
        background: #fef3c7;
        border: 1px solid #fbbf24;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .error-box {
        background: #fee2e2;
        border: 1px solid #f87171;
        border-radius: 8px;
        padding: 15px;
        margin: 15px 0;
    }
    .standalone-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
    }
    .normalized-box {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 15px 0;
    }
    .bank-cell {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# TEMPLATE LOADING
# =============================================================================
@st.cache_data
def load_template():
    """Load HTML template from external file"""
    template_path = Path(__file__).parent / TEMPLATE_FILE

    if not template_path.exists():
        return None, f"Template file not found: {TEMPLATE_FILE}"

    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read(), None
    except Exception as e:
        return None, f"Error loading template: {str(e)}"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def format_rm(value):
    """Format number as RM currency"""
    if value is None or value == 0:
        return "RM 0"
    return f"RM {value:,.0f}"

def _build_fy_config(fy_data, default_period):
    """Build a standard FY config dict from JSON data"""
    return {
        'period': fy_data.get('period', default_period),
        'revenue': fy_data.get('revenue', 0),
        'cos': fy_data.get('cos', 0),
        'gp': fy_data.get('gp', 0),
        'opex': fy_data.get('opex', 0),
        'ebitda': fy_data.get('ebitda', 0),
        'depreciation': fy_data.get('depreciation', 0),
        'finCosts': fy_data.get('finCosts', 0),
        'pbt': fy_data.get('pbt', 0),
        'tax': fy_data.get('tax', 0),
        'netProfit': fy_data.get('netProfit', 0)
    }

def detect_json_version(data):
    """Detect JSON version (v5.1, v5.2, or v5.3)"""
    base_year = data.get('baseYear', {})
    if 'fy3' in base_year:
        return 'v5.3'
    if 'existingCommitment' in data and 'monthlyByBank' in data.get('existingCommitment', {}):
        return 'v5.2'
    if 'facilities' in data:
        return 'v5.1'
    return 'unknown'

def convert_v51_to_v53(data):
    """Convert v5.1 JSON to v5.3 structure"""
    facilities = data.get('facilities', {})
    company_facilities = facilities.get('company', [])

    total_monthly = sum(f.get('monthly', 0) for f in company_facilities)

    return {
        'company': data.get('company', {}),
        'baseYear': data.get('baseYear', {}),
        'existingCommitment': {
            'monthlyByBank': {
                'rhb': total_monthly, 'maybank': total_monthly, 'cimb': total_monthly,
                'stanchart': total_monthly, 'smebank': total_monthly, 'brakyat': total_monthly
            },
            'normalized': total_monthly,
            'normalizedMethod': 'maximum',
            'totalFacilities': len(company_facilities),
            'totalOutstanding': sum(f.get('outstanding', 0) for f in company_facilities)
        },
        'companyFacilities': company_facilities,
        'directorsFacilities': facilities.get('directors', []),
        'pendingApplications': [],
        'nlciCompany': [],
        'projectionDefaults': data.get('projectionDefaults', {}),
        'settings': data.get('settings', {}),
        'metadata': data.get('metadata', {})
    }

def convert_v52_to_v53(data):
    """Convert v5.2 JSON (mgmt) to v5.3 structure (fy3)"""
    base_year = data.get('baseYear', {})
    mgmt = base_year.get('mgmt', {})

    # If mgmt has data, convert it to fy3; otherwise leave fy3 empty
    if mgmt and mgmt.get('revenue', 0) > 0:
        base_year['fy3'] = mgmt
    else:
        base_year['fy3'] = {
            'period': 'FY3', 'revenue': 0, 'cos': 0, 'gp': 0, 'opex': 0,
            'ebitda': 0, 'depreciation': 0, 'finCosts': 0, 'pbt': 0, 'tax': 0, 'netProfit': 0
        }

    # Remove mgmt key
    base_year.pop('mgmt', None)
    data['baseYear'] = base_year
    return data

def generate_config_js(data):
    """Generate JavaScript CONFIG object from JSON data"""
    company = data.get('company', {})
    base_year = data.get('baseYear', {})
    existing = data.get('existingCommitment', {})
    monthly_by_bank = existing.get('monthlyByBank', {})
    presets = data.get('projectionDefaults', {})
    bank_thresholds = data.get('bankThresholds', {})

    fy1 = base_year.get('fy1', base_year.get('fy2023', {}))
    fy2 = base_year.get('fy2', base_year.get('fy2024', {}))
    fy3 = base_year.get('fy3', base_year.get('fy2025', {})) or {}

    company_facilities = data.get('companyFacilities', [])
    directors_facilities = data.get('directorsFacilities', [])

    # Build bank list with thresholds from JSON if available
    default_banks = [
        {'code': 'rhb', 'name': 'RHB Bank', 'minDscr': bank_thresholds.get('rhb', 1.25)},
        {'code': 'mbb', 'name': 'Maybank', 'minDscr': bank_thresholds.get('maybank', 1.25)},
        {'code': 'cimb', 'name': 'CIMB Bank', 'minDscr': bank_thresholds.get('cimb', 1.25)},
        {'code': 'sc', 'name': 'Standard Chartered', 'minDscr': bank_thresholds.get('stanchart', 1.30)},
        {'code': 'sme', 'name': 'SME Bank', 'minDscr': bank_thresholds.get('smebank', 1.25)},
        {'code': 'br', 'name': 'Bank Rakyat', 'minDscr': bank_thresholds.get('brakyat', 1.25)}
    ]

    config = {
        'companyName': company.get('name', 'Company Name'),
        'regNo': company.get('regNo', ''),
        'regNoOld': company.get('regNoOld', ''),
        'businessNature': company.get('businessNature', 'services'),
        'baseYear': {
            'fy1': _build_fy_config(fy1, 'FY1'),
            'fy2': _build_fy_config(fy2, 'FY2'),
            'fy3': _build_fy_config(fy3, 'FY3')
        },
        'companyFacilities': [
            {'no': i+1, 'facility': f.get('facility',''), 'bank': f.get('bank',''),
             'limit': f.get('limit',0), 'outstanding': f.get('outstanding',0), 'monthly': f.get('monthly',0)}
            for i, f in enumerate(company_facilities)
        ],
        'existingCommitment': {
            'monthlyByBank': {
                'rhb': monthly_by_bank.get('rhb', 0),
                'maybank': monthly_by_bank.get('maybank', 0),
                'cimb': monthly_by_bank.get('cimb', 0),
                'stanchart': monthly_by_bank.get('stanchart', 0),
                'smebank': monthly_by_bank.get('smebank', 0),
                'brakyat': monthly_by_bank.get('brakyat', 0)
            },
            'normalized': existing.get('normalized', 0),
            'normalizedMethod': existing.get('normalizedMethod', 'maximum')
        },
        'pendingApplications': data.get('pendingApplications', []),
        'nlciCompany': data.get('nlciCompany', []),
        'directorsFacilities': [
            {'no': i+1, 'facility': f.get('facility',''), 'bank': f.get('bank',''),
             'limit': f.get('limit',0), 'outstanding': f.get('outstanding',0), 'monthly': f.get('monthly',0),
             'director': f.get('director','')}
            for i, f in enumerate(directors_facilities)
        ],
        'existingWcFacility': data.get('facilities', {}).get('existingWcFacility', 0),
        'banks': default_banks,
        'presets': {
            'conservative': presets.get('conservative', {
                'sg': [5,3,2], 'em': [8,8,8], 'cos': [65,65,65], 'opex': [25,25,25],
                'deb': [30,30,30], 'stk': [0,0,0], 'cred': [30,30,30]
            }),
            'base': presets.get('base', {
                'sg': [10,8,5], 'em': [10,10,10], 'cos': [62,62,62], 'opex': [25,25,25],
                'deb': [30,30,30], 'stk': [0,0,0], 'cred': [30,30,30]
            }),
            'aggressive': presets.get('aggressive', {
                'sg': [20,15,12], 'em': [12,13,14], 'cos': [58,56,55], 'opex': [22,20,18],
                'deb': [30,30,30], 'stk': [0,0,0], 'cred': [30,30,30]
            })
        },
        'sensitivity': [
            {'name': 'Worst Case', 'sg': -10, 'em': 5},
            {'name': 'Conservative', 'sg': 10, 'em': 8},
            {'name': 'Base Case', 'sg': 'current', 'em': 'current'},
            {'name': 'Aggressive', 'sg': 75, 'em': 15}
        ]
    }

    return json.dumps(config, indent=2)

def generate_html(data, template):
    """Generate complete HTML from JSON data and template"""
    config_js = generate_config_js(data)

    # Replace CONFIG object
    pattern = r'const CONFIG = \{[\s\S]*?\n  \};'
    replacement = f'const CONFIG = {config_js};'
    html = re.sub(pattern, replacement, template)

    # Replace placeholders
    company = data.get('company', {})
    html = html.replace('[COMPANY_NAME]', company.get('name', 'Company Name'))
    html = html.replace('[REG_NO]', company.get('regNo', ''))

    return html

# =============================================================================
# MAIN APP
# =============================================================================
def main():
    st.markdown(f"""
    <div class="main-header">
        <h1>📊 DSCR Calculator Generator <span class="standalone-badge">v{VERSION}</span></h1>
        <p>Upload JSON → Generate interactive HTML calculator (Option C - Normalized, 3 Audited Years)</p>
    </div>
    """, unsafe_allow_html=True)

    # Load template
    template, error = load_template()

    if error:
        st.markdown(f"""
        <div class="error-box">
            ❌ <strong>Template Error:</strong> {error}<br>
            <small>Make sure <code>{TEMPLATE_FILE}</code> is in the same directory as this script.</small>
        </div>
        """, unsafe_allow_html=True)
        return

    uploaded_file = st.file_uploader(
        "Upload JSON file (supports v5.1, v5.2 and v5.3 formats)",
        type=['json'],
        help="Upload JSON with company financial data"
    )

    if uploaded_file is not None:
        try:
            raw_data = json.load(uploaded_file)
            version = detect_json_version(raw_data)

            if version == 'v5.1':
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>v5.1 JSON detected!</strong> Converting to v5.3 format...
                </div>
                """, unsafe_allow_html=True)
                data = convert_v51_to_v53(raw_data)
            elif version == 'v5.2':
                st.markdown("""
                <div class="warning-box">
                    ⚠️ <strong>v5.2 JSON detected!</strong> Converting mgmt → fy3...
                </div>
                """, unsafe_allow_html=True)
                data = convert_v52_to_v53(raw_data)
            elif version == 'v5.3':
                data = raw_data
                st.markdown("""
                <div class="success-box">
                    ✅ <strong>v5.3 JSON loaded successfully!</strong> (3 audited years detected)
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Unknown JSON format")
                return

            # Display summary
            company = data.get('company', {})
            base_year = data.get('baseYear', {})
            existing = data.get('existingCommitment', {})

            st.markdown("### 🏢 Company Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.text_input("Company Name", company.get('name', ''), disabled=True)
            with col2:
                st.text_input("Registration No", company.get('regNo', ''), disabled=True)
            with col3:
                st.text_input("Business Nature", company.get('businessNature', ''), disabled=True)

            # Show all FY summaries
            st.markdown("### 📈 Financial Summary (All Audited Years)")
            fy_keys = ['fy1', 'fy2', 'fy3']
            fy_data_list = []
            for key in fy_keys:
                fy = base_year.get(key, {})
                if fy and fy.get('revenue', 0) > 0:
                    fy_data_list.append((key, fy))

            if fy_data_list:
                cols = st.columns(len(fy_data_list))
                for i, (key, fy) in enumerate(fy_data_list):
                    with cols[i]:
                        period = fy.get('period', key.upper())
                        revenue = fy.get('revenue', 0)
                        ebitda = fy.get('ebitda', 0)
                        margin = (ebitda / revenue * 100) if revenue > 0 else 0
                        st.markdown(f"**{period}**")
                        st.metric("Revenue", format_rm(revenue))
                        st.metric("EBITDA", format_rm(ebitda))
                        st.metric("EBITDA Margin", f"{margin:.1f}%")

            # Normalized commitment
            normalized = existing.get('normalized', 0)
            st.metric("Normalized Monthly Commitment", format_rm(normalized) + "/mo")

            # Bank breakdown
            if 'monthlyByBank' in existing:
                st.markdown("### 🏦 Existing Commitment (Normalized)")
                st.markdown(f"""
                <div class="normalized-box">
                    <div style="font-size:12px;">Normalized Monthly (Maximum across 6 banks)</div>
                    <div style="font-size:32px;font-weight:700;">{format_rm(normalized)}</div>
                </div>
                """, unsafe_allow_html=True)

                monthly_by_bank = existing.get('monthlyByBank', {})
                cols = st.columns(6)
                banks = [('RHB', 'rhb'), ('MBB', 'maybank'), ('CIMB', 'cimb'),
                         ('SC', 'stanchart'), ('SME', 'smebank'), ('BR', 'brakyat')]
                for i, (name, code) in enumerate(banks):
                    with cols[i]:
                        val = monthly_by_bank.get(code, 0)
                        is_max = val == normalized and val > 0
                        style = "background:#dcfce7;" if is_max else ""
                        st.markdown(f'<div class="bank-cell" style="{style}"><b>{name}</b><br>{format_rm(val)}</div>',
                                   unsafe_allow_html=True)

            # Generate button
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("🔧 Generate HTML Calculator", type="primary", use_container_width=True):
                    with st.spinner("Generating..."):
                        html = generate_html(data, template)
                        st.session_state['html'] = html
                        st.success("��� HTML generated!")

            with col2:
                st.download_button(
                    "📥 Download v5.3 JSON",
                    json.dumps(data, indent=2),
                    f"DSCR_{company.get('shortName','DATA')}_v5.3.json",
                    "application/json",
                    use_container_width=True
                )

            if 'html' in st.session_state:
                st.markdown("---")
                filename = f"DSCR_Calculator_{company.get('shortName', 'COMPANY')}_{datetime.now().strftime('%Y%m%d')}.html"
                st.download_button(
                    "📥 Download HTML Calculator",
                    st.session_state['html'],
                    filename,
                    "text/html",
                    type="primary",
                    use_container_width=True
                )
                st.info(f"📁 File: **{filename}**")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    else:
        st.markdown("""
        <div class="preview-card">
            <h3>📋 How to Use</h3>
            <ol>
                <li>Get JSON from Claude (v5.1, v5.2 or v5.3 format)</li>
                <li>Upload JSON using the uploader above</li>
                <li>Review the extracted data</li>
                <li>Click Generate to create HTML</li>
                <li>Download and open in browser</li>
            </ol>
            <p style="margin-top:15px;color:#64748b;">
                <strong>v5.3 Features:</strong> 3 audited years (fy1/fy2/fy3), normalized commitment, base year selector, dual theme (light/dark mode)
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
