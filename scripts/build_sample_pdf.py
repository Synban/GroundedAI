"""Create a tiny redistributable syllabus PDF for the starter corpus."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "pdfs" / "sample_syllabus.pdf"

LINES = [
    "CS4320 Sample Course Syllabus",
    "",
    "Instructor: Dr. Ada Example",
    "Office hours: Tuesdays 2:00 PM to 3:00 PM in Room 204.",
    "",
    "Grading policy:",
    "Homework is worth 40 percent of the final grade.",
    "The midterm exam is worth 25 percent.",
    "The final project is worth 35 percent.",
    "",
    "Late work policy:",
    "Assignments may be submitted up to 48 hours late with a 10 percent penalty.",
    "Work submitted more than 48 hours late receives no credit unless an extension was approved.",
    "",
    "Project requirement:",
    "Students must solve a clearly defined machine learning problem.",
    "Kaggle competition submissions are not allowed for the course project.",
    "Repositories must be reproducible and include evaluation.",
]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_content_stream(lines: list[str]) -> str:
    commands = ["BT", "/F1 12 Tf", "72 720 Td", "16 TL"]
    for i, line in enumerate(lines):
        if i == 0:
            commands.append(f"({escape_pdf_text(line)}) Tj")
        else:
            commands.append("T*")
            commands.append(f"({escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands)


def write_pdf(path: Path) -> None:
    stream = build_content_stream(LINES)
    stream_bytes = stream.encode("latin-1", errors="replace")

    objects: list[bytes] = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objects.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objects.append(
        f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
        + stream_bytes
        + b"\nendstream"
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n".encode("ascii")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(out)
    print(f"Wrote {path}")


if __name__ == "__main__":
    write_pdf(OUT)
