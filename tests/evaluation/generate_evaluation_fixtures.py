"""Generate synthetic PDF fixtures for Phase 7 deterministic evaluation."""

from pathlib import Path
import pymupdf
from PIL import Image, ImageDraw, ImageFont
import io


def create_evaluation_fixtures(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Conflicting Policy Fixture
    conflict_path = output_dir / "conflicting_policy_fixture.pdf"
    doc_conflict = pymupdf.open()

    p1 = doc_conflict.new_page(width=595, height=842)
    p1_text = (
        "CONFLICTING HEALTH INSURANCE POLICY\n"
        "Section 1: General Policy Terms\n"
        "Sum Insured: Rs. 5,00,000. Entry age: 18 to 65 years.\n\n"
        "1.1 Pre-existing Disease Waiting Period: 24 months of continuous coverage before claims are admissible.\n"
        "1.2 Co-payment: A mandatory 10% co-payment applies to all admissible hospitalisation claims.\n"
        "1.3 Room Rent: Up to Rs. 3,000 per day.\n"
    )
    p1.insert_textbox(pymupdf.Rect(50, 50, 545, 400), p1_text, fontsize=11)

    p2 = doc_conflict.new_page(width=595, height=842)
    p2_text = (
        "Section 2: Special Provisions and Endorsement Clauses\n"
        "Notwithstanding Section 1, the following special conditions apply:\n\n"
        "2.1 Pre-existing Disease Waiting Period: 48 months of continuous coverage is strictly required.\n"
        "2.2 Co-payment: 20% co-payment shall apply to all claims for all insured persons.\n"
        "2.3 Specific Illness Waiting Period: 24 months for specified surgeries.\n"
    )
    p2.insert_textbox(pymupdf.Rect(50, 50, 545, 400), p2_text, fontsize=11)
    doc_conflict.save(str(conflict_path))
    doc_conflict.close()
    print(f"Created: {conflict_path}")

    # 2. Explicit Absence Policy Fixture
    absence_path = output_dir / "explicit_absence_policy_fixture.pdf"
    doc_absence = pymupdf.open()

    p1 = doc_absence.new_page(width=595, height=842)
    p1_text = (
        "ZERO RESTRICTION HEALTH POLICY\n"
        "Section 1: Policy Benefits and Waivers\n"
        "Sum Insured: Rs. 10,00,000. Entry age: 18 to 65 years.\n\n"
        "1.1 Room Rent Limit: No room-rent capping or sub-limit applies under this policy. Any room category allowed without restriction.\n"
        "1.2 Co-payment: No co-payment is applicable under this policy. 0% co-payment applies to all eligible claims.\n"
        "1.3 Deductible: There is no deductible applicable under this policy. Compulsory deductible is nil.\n"
    )
    p1.insert_textbox(pymupdf.Rect(50, 50, 545, 400), p1_text, fontsize=11)

    p2 = doc_absence.new_page(width=595, height=842)
    p2_text = (
        "Section 2: Waiting Periods and Exclusions\n"
        "2.1 Initial Waiting Period: 30 days from inception for illnesses.\n"
        "2.2 Specific Illness Waiting Period: 24 months for cataract and hernia.\n"
        "2.3 Pre-existing Disease Waiting Period: 36 months of continuous coverage.\n"
        "2.4 Permanent Exclusions: Hazardous sports and self-inflicted injury.\n"
    )
    p2.insert_textbox(pymupdf.Rect(50, 50, 545, 400), p2_text, fontsize=11)
    doc_absence.save(str(absence_path))
    doc_absence.close()
    print(f"Created: {absence_path}")

    # 3. Scanned Image Policy Fixture (zero native text stream)
    scanned_path = output_dir / "scanned_image_policy_fixture.pdf"
    img = Image.new("RGB", (800, 1100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((60, 60), "SCANNED IMAGE ONLY POLICY DOCUMENT", fill=(0, 0, 0))
    draw.text((60, 100), "This document has no native digital text stream.", fill=(0, 0, 0))
    draw.text((60, 140), "Pre-existing Disease Waiting Period: 36 months.", fill=(0, 0, 0))
    draw.text((60, 180), "Co-payment: 10% co-payment applies.", fill=(0, 0, 0))

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()

    doc_scanned = pymupdf.open()
    page = doc_scanned.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=img_bytes)
    doc_scanned.save(str(scanned_path))
    doc_scanned.close()
    print(f"Created: {scanned_path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "fixtures"
    create_evaluation_fixtures(out)
