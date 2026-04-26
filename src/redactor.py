"""
PII Redactor Module

Uses PyMuPDF to detect and redact personal identifiable information (PII)
from bank statement PDFs, including names, credit card numbers, account
numbers, addresses, phone numbers, and email addresses.
"""

import io
import re

import fitz  # PyMuPDF


class PiiRedactor:
    """
    Detects and redacts personal identifiable information from PDF documents.

    Targets the following PII types commonly found in bank statements:
        - Credit/debit card numbers (16-digit sequences)
        - Masked card numbers (e.g. ****-****-****-1234)
        - Bank account and reference numbers (near identifying keywords)
        - Personal names (via context keywords and address block detection)
        - Mailing addresses (street, city, province, postal code)
        - Phone numbers
        - Email addresses

    Usage:
        redactor = PiiRedactor()
        redacted_bytes = redactor.redact(original_pdf_bytes)
    """

    # ── Direct pattern matching ──────────────────────────────────────────

    # Credit/debit card: 4 groups of 4 digits
    CARD_PATTERN = re.compile(
        r'\b(\d{4}[\s\-]{0,2}\d{4}[\s\-]{0,2}\d{4}[\s\-]{0,2}\d{4})\b'
    )

    # Masked card numbers: ****-****-****-1234 variants
    MASKED_CARD_PATTERN = re.compile(
        r'[*Xx]{4}[\s\-]{0,2}[*Xx]{4}[\s\-]{0,2}[*Xx]{4}[\s\-]{0,2}\d{4}'
    )

    # Partially masked card: 5598 28** **** 8007 (digits + asterisks mixed)
    PARTIAL_MASKED_CARD_PATTERN = re.compile(
        r'\b\d{4}[\s\-]{0,2}\d{0,2}[*]{2,4}[\s\-]{0,2}[*]{4}[\s\-]{0,2}\d{4}\b'
    )

    # Phone: (XXX) XXX-XXXX or XXX-XXX-XXXX or XXX.XXX.XXXX
    PHONE_PATTERN = re.compile(
        r'\(?\b\d{3}\)?[\s\-\.]{1,2}\d{3}[\s\-\.]\d{4}\b'
    )

    # Email addresses
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'
    )

    # Canadian postal code: A1A 1A1
    POSTAL_CODE_PATTERN = re.compile(
        r'\b([A-Za-z]\d[A-Za-z][\s\-]?\d[A-Za-z]\d)\b'
    )

    # ── Keyword-proximity patterns ───────────────────────────────────────

    # Account/card numbers near identifying keywords (EN + FR)
    ACCOUNT_PATTERNS = [
        re.compile(
            r'(?:Account|Compte|Carte|Card)'
            r'[\s]*(?:Number|No\.?|#|number|num[ée]ro|de compte|de carte)?'
            r'[\s:]*'
            r'([\d][\d\s\-\.]{4,24}[\d])',
            re.IGNORECASE,
        ),
        re.compile(
            r'(?:Num[ée]ro)'
            r'[\s]*(?:de\s+)?(?:compte|carte|r[ée]f[ée]rence)'
            r'[\s:]*'
            r'([\d][\d\s\-\.]{4,24}[\d])',
            re.IGNORECASE,
        ),
    ]

    # Name near context keywords (captures to end of line)
    NAME_PATTERNS = [
        re.compile(
            r'(?:'
            r'Account\s*Holder|Card\s*(?:Member|Holder)|'
            r'Nom\s*(?:du\s*)?(?:titulaire|client|d[ée]tenteur|membre)|'
            r'Titulaire|'
            r'Monsieur|Madame|M\.|Mme|Mr\.?|Mrs\.?|Ms\.?'
            r')'
            r'[\s:,]+(.+?)$',
            re.IGNORECASE | re.MULTILINE,
        ),
        # French: "carte de : NAME" / "effectuées ... carte de : NAME"
        re.compile(
            r'(?:carte\s+de|carte\s+de\s+cr[ée]dit\s+de)'
            r'[\s:]+'
            r'([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F\-\']+'
            r'(?:\s+[A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F\-\']+)+)',
            re.IGNORECASE,
        ),
    ]

    # Street address: civic number + street name + street type
    ADDRESS_PATTERN = re.compile(
        r'\b(\d{1,5}\s+'
        r'(?:[\w\'\u00C0-\u024F\-]+\s+){1,5}'
        r'(?:Street|St\.?|Avenue|Ave\.?|Boulevard|Blvd\.?|Road|Rd\.?|'
        r'Drive|Dr\.?|Lane|Ln\.?|Court|Ct\.?|Place|Pl\.?|Way|'
        r'Rue|Chemin|Ch\.?|Boul\.?|Av\.?|Cres\.?|Crescent|'
        r'Circle|Cir\.?|Terrace|Terr\.?|Trail|Trl\.?|'
        r'Parkway|Pkwy\.?|Highway|Hwy\.?)'
        r'(?:\.|\b))',
        re.IGNORECASE,
    )

    # City + Province/Territory (Canadian)
    CITY_PROV_PATTERN = re.compile(
        r'([\w\u00C0-\u024F]+(?:[\s\-][\w\u00C0-\u024F]+)*)'
        r'[\s,]+'
        r'(QC|ON|BC|AB|MB|SK|NB|NS|PE|NL|NT|NU|YT|'
        r'Qu[ée]bec|Ontario|British\s+Columbia|Alberta|Manitoba|'
        r'Saskatchewan|New\s+Brunswick|Nova\s+Scotia|'
        r'Prince\s+Edward\s+Island|Newfoundland(?:\s+and\s+Labrador)?)'
        r'(?:[\s,]+[A-Za-z]\d[A-Za-z][\s\-]?\d[A-Za-z]\d)?',
        re.IGNORECASE,
    )

    REDACTION_FILL = (0, 0, 0)  # Black fill for redacted areas

    def redact(self, pdf_bytes: bytes) -> bytes:
        """
        Redact all detected PII from a PDF document.

        Args:
            pdf_bytes: Raw bytes of the original PDF.

        Returns:
            Bytes of the redacted PDF with PII replaced by black rectangles.
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Phase 1: scan entire document for names (they may repeat across pages)
        full_text = "\n".join(page.get_text() for page in doc)
        known_names = self._extract_names(full_text)

        # Phase 1b: extract names from address blocks (name above postal code)
        if len(doc) > 0:
            self._extract_names_from_address_blocks(doc[0], known_names)

        # Phase 2: redact each page
        for page_num, page in enumerate(doc):
            self._redact_page(page, known_names, is_first_page=(page_num == 0))

        output = io.BytesIO()
        doc.save(output, garbage=4, deflate=True)
        doc.close()
        return output.getvalue()

    def redact_file(self, input_path: str, output_path: str) -> None:
        """
        Redact PII from a PDF file and save the result.

        Args:
            input_path: Path to the original PDF.
            output_path: Path to save the redacted PDF.
        """
        with open(input_path, "rb") as f:
            pdf_bytes = f.read()
        redacted = self.redact(pdf_bytes)
        with open(output_path, "wb") as f:
            f.write(redacted)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _extract_names(self, text: str) -> set[str]:
        """Scan full document text for personal names using context keywords."""
        names: set[str] = set()
        for pattern in self.NAME_PATTERNS:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if name and len(name) >= 3 and not name.isdigit():
                    self._add_name_variants(names, name)
        return names

    def _extract_names_from_address_blocks(
        self, page: fitz.Page, names: set[str]
    ) -> None:
        """
        Extract names from address blocks on the given page.

        In bank statements the recipient name is typically the first line
        of the mailing address block (the block that contains a postal code).
        """
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue

            lines = block.get("lines", [])
            block_text = " ".join(
                span["text"]
                for line in lines
                for span in line.get("spans", [])
            )

            if not self.POSTAL_CODE_PATTERN.search(block_text):
                continue

            # The first line of an address block is usually the name
            if lines:
                first_line = " ".join(
                    span["text"] for span in lines[0].get("spans", [])
                ).strip()
                # Validate: must look like a name (letters, spaces, hyphens)
                if (
                    first_line
                    and len(first_line) >= 3
                    and not first_line[0].isdigit()
                    and not self.POSTAL_CODE_PATTERN.match(first_line)
                ):
                    self._add_name_variants(names, first_line)

    @staticmethod
    def _add_name_variants(names: set[str], name: str) -> None:
        """Add a name and its common case variants to the set."""
        names.add(name)
        names.add(name.upper())
        names.add(name.title())
        # Also add with extra whitespace collapsed
        collapsed = " ".join(name.split())
        if collapsed != name:
            names.add(collapsed)
            names.add(collapsed.upper())
            names.add(collapsed.title())

    def _redact_page(
        self,
        page: fitz.Page,
        known_names: set[str],
        *,
        is_first_page: bool = False,
    ) -> None:
        """Identify and redact all PII on a single page."""
        text = page.get_text()
        if not text:
            return

        pii_texts: set[str] = set()

        # Collect pattern-based PII
        self._collect_pattern_matches(text, pii_texts)

        # Add document-wide known names
        pii_texts.update(known_names)

        # Annotate each PII occurrence for redaction
        has_redactions = False
        for pii_text in pii_texts:
            for rect in page.search_for(pii_text):
                page.add_redact_annot(rect, fill=self.REDACTION_FILL)
                has_redactions = True

        # On the first page, also redact the mailing address block
        if is_first_page:
            if self._redact_address_block(page):
                has_redactions = True

        if has_redactions:
            page.apply_redactions()

    def _redact_address_block(self, page: fitz.Page) -> bool:
        """
        Detect and redact the mailing address block on the page.

        Bank statements typically place the recipient's name and mailing
        address together in a single text block. This method finds blocks
        containing a Canadian postal code and redacts the entire block,
        covering the name, street address, and city/province/postal code
        lines as a unit.

        Returns:
            True if any block was redacted.
        """
        redacted = False
        text_dict = page.get_text("dict")

        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # text blocks only
                continue

            block_text = " ".join(
                span["text"]
                for line in block.get("lines", [])
                for span in line.get("spans", [])
            )

            if self.POSTAL_CODE_PATTERN.search(block_text):
                rect = fitz.Rect(block["bbox"])
                page.add_redact_annot(rect, fill=self.REDACTION_FILL)
                redacted = True

        return redacted

    def _collect_pattern_matches(self, text: str, results: set[str]) -> None:
        """Run all PII regex patterns against page text and collect matches."""
        # Credit/debit card numbers
        for match in self.CARD_PATTERN.finditer(text):
            results.add(match.group(1).strip())

        # Masked card numbers
        for match in self.MASKED_CARD_PATTERN.finditer(text):
            results.add(match.group().strip())

        # Partially masked card numbers (e.g. 5598 28** **** 8007)
        for match in self.PARTIAL_MASKED_CARD_PATTERN.finditer(text):
            results.add(match.group().strip())

        # Phone numbers
        for match in self.PHONE_PATTERN.finditer(text):
            results.add(match.group().strip())

        # Email addresses
        for match in self.EMAIL_PATTERN.finditer(text):
            results.add(match.group().strip())

        # Postal codes
        for match in self.POSTAL_CODE_PATTERN.finditer(text):
            results.add(match.group(1).strip())

        # Account numbers near keywords
        for pattern in self.ACCOUNT_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                # Require at least 4 digits to avoid false positives
                if value and len(re.sub(r'\D', '', value)) >= 4:
                    results.add(value)

        # Street addresses
        for match in self.ADDRESS_PATTERN.finditer(text):
            results.add(match.group(1).strip())

        # City + Province lines
        for match in self.CITY_PROV_PATTERN.finditer(text):
            results.add(match.group().strip())
