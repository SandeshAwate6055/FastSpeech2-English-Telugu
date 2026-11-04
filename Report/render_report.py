from pathlib import Path

import pypdfium2 as pdfium


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "Report" / "preview"
    destination.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(str(root / "Report" / "Technical_Report.pdf"))
    for index in range(len(document)):
        page = document[index]
        bitmap = page.render(scale=1.3)
        image = bitmap.to_pil()
        image.save(destination / f"page-{index + 1:02d}.png")
        bitmap.close()
        page.close()
    print(f"Rendered {len(document)} pages to {destination}")
    document.close()


if __name__ == "__main__":
    main()
