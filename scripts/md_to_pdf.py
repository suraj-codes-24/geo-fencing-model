import os
from markdown_pdf import MarkdownPdf, Section

md_file = "docs/architecture_spec.md"
pdf_file = "docs/architecture_spec.pdf"

if os.path.exists(md_file):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
        
    pdf = MarkdownPdf()
    pdf.add_section(Section(md_content))
    pdf.save(pdf_file)
    print("PDF saved successfully!")
else:
    print("Markdown file not found.")
