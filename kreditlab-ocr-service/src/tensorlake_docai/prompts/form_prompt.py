# SPDX-License-Identifier: Apache-2.0
FIGURE_FORM_GENERIC_PROMPT = (
    "Extract all field-value pairs from this form.\n\n"
    "Format:\n"
    "field_name: value\n\n"
    "Rules:\n"
    "- One field per line\n"
    "- Use field label exactly as shown\n"
    "- For checkboxes: checked/unchecked or the selected option\n"
    "- For empty fields: empty\n"
    "- For multi-line values: join with semicolon\n\n"
    "Example:\n"
    "patient_name: Carter, Jamie\n"
    "date_of_birth: 04/22/1990\n"
    "insurance_type: FECA\n"
)

FIGURE_FORM_TEMPLATE_PROMPT = (
    "Extract form data matching these fields:\n\n"
    "{field_list}\n\n"
    "Output as JSON. Use null for empty fields."
)

# Form type detection - updated with your 10 forms
FORM_TYPE_DETECTION_PROMPT = (
    "Identify the specific form type. Choose ONE:\n"
    "TAX FORMS:\n"
    "- 1040: U.S. Individual Income Tax Return\n"
    "- W2: Wage and Tax Statement\n"
    "- 1099_NEC: Nonemployee Compensation\n"
    "- 1099_MISC: Miscellaneous Income\n\n"
    "MEDICAL FORMS:\n"
    "- CMS_1500: Health Insurance Claim Form\n"
    "- UB_04: Hospital/Facility Claim Form (CMS-1450)\n\n"
    "EMPLOYMENT FORMS:\n"
    "- I9: Employment Eligibility Verification\n"
    "- W4: Employee's Withholding Certificate\n\n"
    "PASSPORT FORMS:\n"
    "- DS_11: U.S. Passport Application\n"
    "- DS_82: U.S. Passport Renewal\n\n"
    "- UNKNOWN: Any other form\n\n"
    "Output ONLY the type name (e.g., '1040'). No explanation."
)

# Form templates - key fields for each form type
FORM_TEMPLATES = {
    # TAX FORMS
    "1040": [
        "tax_year",
        "filing_status",
        "first_name",
        "last_name",
        "social_security_number",
        "spouse_first_name",
        "spouse_last_name",
        "spouse_ssn",
        "address",
        "city_state_zip",
        "wages_salaries_tips",
        "taxable_interest",
        "qualified_dividends",
        "capital_gain_loss",
        "total_income",
        "adjusted_gross_income",
        "standard_deduction",
        "taxable_income",
        "total_tax",
        "federal_income_tax_withheld",
        "refund_amount",
        "amount_owed",
    ],
    "W2": [
        "tax_year",
        "employer_name",
        "employer_ein",
        "employer_address",
        "employee_name",
        "employee_ssn",
        "employee_address",
        "wages_tips_compensation",
        "federal_income_tax_withheld",
        "social_security_wages",
        "social_security_tax_withheld",
        "medicare_wages_tips",
        "medicare_tax_withheld",
        "state",
        "state_wages_tips",
        "state_income_tax",
    ],
    "1099_NEC": [
        "tax_year",
        "payer_name",
        "payer_tin",
        "payer_address",
        "recipient_name",
        "recipient_tin",
        "recipient_address",
        "nonemployee_compensation",
        "state_tax_withheld",
        "state",
        "state_income",
    ],
    "1099_MISC": [
        "tax_year",
        "payer_name",
        "payer_tin",
        "payer_address",
        "recipient_name",
        "recipient_tin",
        "recipient_address",
        "rents",
        "royalties",
        "other_income",
        "federal_income_tax_withheld",
        "fishing_boat_proceeds",
        "medical_health_payments",
        "substitute_payments",
        "crop_insurance_proceeds",
        "state_tax_withheld",
        "state",
        "state_income",
    ],
    # MEDICAL FORMS
    "CMS_1500": [
        "insurance_type",
        "insured_id_number",
        "patient_name",
        "patient_birth_date",
        "patient_sex",
        "patient_address",
        "patient_relationship_to_insured",
        "insured_name",
        "insured_address",
        "insurance_plan_name",
        "patient_condition_related_to_employment",
        "patient_condition_related_to_auto_accident",
        "illness_injury_date",
        "referring_provider_name",
        "hospitalization_dates",
        "diagnosis_codes",
        "service_dates",
        "place_of_service",
        "procedure_codes",
        "charges",
        "total_charge",
        "amount_paid",
        "provider_name",
        "provider_npi",
        "facility_name",
        "facility_address",
    ],
    "UB_04": [
        "patient_name",
        "patient_id",
        "patient_birth_date",
        "patient_sex",
        "patient_address",
        "admission_date",
        "discharge_date",
        "patient_status",
        "occurrence_codes",
        "occurrence_dates",
        "value_codes",
        "value_amounts",
        "revenue_codes",
        "service_descriptions",
        "service_dates",
        "units_of_service",
        "total_charges",
        "non_covered_charges",
        "payer_name",
        "insured_name",
        "patient_relationship_to_insured",
        "insurance_group_name",
        "treatment_authorization_codes",
        "provider_name",
        "provider_npi",
        "attending_physician_name",
        "attending_physician_npi",
    ],
    # EMPLOYMENT FORMS
    "I9": [
        "employee_last_name",
        "employee_first_name",
        "employee_middle_initial",
        "other_names_used",
        "employee_address",
        "employee_birth_date",
        "employee_ssn",
        "employee_email",
        "employee_phone",
        "citizenship_status",
        "alien_registration_number",
        "form_i94_number",
        "foreign_passport_number",
        "country_of_issuance",
        "employee_signature_date",
        "preparer_translator_used",
        "preparer_last_name",
        "preparer_first_name",
        "employer_name",
        "employer_first_day_of_employment",
        "document_title",
        "issuing_authority",
        "document_number",
        "expiration_date",
        "employer_signature_date",
    ],
    "W4": [
        "tax_year",
        "employee_first_name",
        "employee_last_name",
        "employee_address",
        "employee_city_state_zip",
        "employee_ssn",
        "filing_status",
        "multiple_jobs_or_spouse_works",
        "claim_dependents",
        "dependents_amount",
        "other_income",
        "deductions",
        "extra_withholding",
        "exempt_from_withholding",
        "employee_signature",
        "employee_signature_date",
        "employer_name",
        "employer_ein",
        "employer_address",
    ],
    # PASSPORT FORMS
    "DS_11": [
        "applicant_last_name",
        "applicant_first_name",
        "applicant_middle_name",
        "date_of_birth",
        "sex",
        "place_of_birth_city",
        "place_of_birth_state",
        "place_of_birth_country",
        "ssn",
        "email",
        "mailing_address",
        "phone_number",
        "height",
        "hair_color",
        "eye_color",
        "occupation",
        "employer",
        "travel_plans",
        "departure_date",
        "father_last_name",
        "father_first_name",
        "father_date_of_birth",
        "father_place_of_birth",
        "mother_last_name",
        "mother_first_name",
        "mother_date_of_birth",
        "mother_place_of_birth",
        "emergency_contact_name",
        "emergency_contact_address",
        "emergency_contact_phone",
    ],
    "DS_82": [
        "applicant_last_name",
        "applicant_first_name",
        "applicant_middle_name",
        "date_of_birth",
        "sex",
        "place_of_birth",
        "ssn",
        "email",
        "mailing_address",
        "phone_number",
        "height",
        "hair_color",
        "eye_color",
        "occupation",
        "employer",
        "travel_plans",
        "departure_date",
        "most_recent_passport_number",
        "passport_issue_date",
        "name_change",
        "previous_name",
        "emergency_contact_name",
        "emergency_contact_address",
        "emergency_contact_phone",
    ],
}
