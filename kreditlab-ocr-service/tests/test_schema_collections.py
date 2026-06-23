# SPDX-License-Identifier: Apache-2.0
"""Tests for sample schema models in extraction/schema_collections.py."""

from datetime import date

from tensorlake_docai.extraction.schema_collections import (
    Address,
    BankStatement,
    BankTransaction,
    Receipt,
)


def test_address_defaults_all_none():
    addr = Address()
    assert addr.street is None
    assert addr.city is None
    assert addr.state is None
    assert addr.zip_code is None


def test_address_with_values():
    addr = Address(street="123 Main St", city="Springfield", state="IL", zip_code="62701")
    assert addr.street == "123 Main St"
    assert addr.city == "Springfield"
    assert addr.state == "IL"
    assert addr.zip_code == "62701"


def test_bank_transaction_defaults_all_none():
    txn = BankTransaction()
    assert txn.transaction_deposit is None
    assert txn.transaction_withdrawal is None
    assert txn.transaction_deposit_date is None


def test_bank_transaction_with_values():
    txn = BankTransaction(
        transaction_deposit=500.00,
        transaction_deposit_date=date(2024, 1, 15),
        transaction_deposit_description="Paycheck",
        transaction_withdrawal=100.00,
        transaction_withdrawal_date=date(2024, 1, 20),
        transaction_withdrawal_description="Rent",
    )
    assert txn.transaction_deposit == 500.00
    assert txn.transaction_deposit_date == date(2024, 1, 15)
    assert txn.transaction_withdrawal == 100.00


def test_bank_statement_defaults():
    stmt = BankStatement()
    assert stmt.account_number is None
    assert stmt.bank_name is None
    assert stmt.table_item is None
    assert stmt.others is None


def test_bank_statement_with_nested_models():
    addr = Address(street="1 Bank Plaza", city="Chicago")
    txn = BankTransaction(transaction_deposit=1000.0)
    stmt = BankStatement(
        account_number="12345678",
        account_type="Checking",
        bank_name="First National",
        bank_address=addr,
        starting_balance=5000.0,
        ending_balance=5900.0,
        table_item=[txn],
    )
    assert stmt.account_number == "12345678"
    assert stmt.bank_address.city == "Chicago"
    assert len(stmt.table_item) == 1
    assert stmt.table_item[0].transaction_deposit == 1000.0


def test_receipt_defaults():
    r = Receipt()
    assert r.store is None
    assert r.receipt_date is None
    assert r.total is None


def test_receipt_with_values():
    r = Receipt(store="Whole Foods", receipt_date=date(2024, 3, 10), total=87.42)
    assert r.store == "Whole Foods"
    assert r.total == 87.42


def test_bank_statement_serialise_round_trip():
    stmt = BankStatement(bank_name="Test Bank", starting_balance=100.0)
    data = stmt.model_dump()
    restored = BankStatement(**data)
    assert restored.bank_name == "Test Bank"
    assert restored.starting_balance == 100.0
