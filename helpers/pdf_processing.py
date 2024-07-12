import fitz
import os

def split_pdf_to_images(pdf_filename, output_dir="./data/Images", zoom_x=2.0, zoom_y=2.0):
    mat = fitz.Matrix(zoom_x, zoom_y)
    doc = fitz.open(pdf_filename)
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        outpath = os.path.join(output_dir, f"{pdf_filename}_{page.number}.jpg")
        pix.save(outpath)
    return [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith(".jpg")]
