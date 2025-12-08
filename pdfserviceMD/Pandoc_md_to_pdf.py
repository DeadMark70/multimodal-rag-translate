import os
import tempfile
import pypandoc
import re


def create_temp_markdown_file(markdown_text):
    """Create a temporary markdown file and return its path."""
    temp_md_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".md", mode="w", encoding="utf-8"
    )
    temp_md_file.write(markdown_text)
    temp_md_file.close()
    return temp_md_file.name


def MDmarkdown_to_pdf(markdown_text, output_pdf: str):
    markdownfile = None
    try:
        markdownfile = create_temp_markdown_file(markdown_text)

        extra_pandoc_args = [
            "--pdf-engine=xelatex",
            "-s",
            '--variable', 'documentclass=ctexart',
            '--variable', 'mainfont=Microsoft JhengHei',
            '--variable', 'CJKmainfont=Microsoft JhengHei',
            '--variable', 'monofont=Consolas',
            '--variable', 'mathfont=Latin Modern Math', 
            '--variable', 'geometry=margin=1in',
            '--variable', 'lang=zh-TW',
        ]

        pypandoc.convert_file(
            source_file=markdownfile,
            to="pdf",
            outputfile=output_pdf,
            extra_args=extra_pandoc_args,
        )
        print(f"PDF created successfully: {output_pdf}")

        return output_pdf

    except RuntimeError as e:
        print(f"An error occurred: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    finally:
        if markdownfile and os.path.exists(markdownfile):
            os.remove(markdownfile)
            print(f"Removed temporary Markdown file: {markdownfile}")
