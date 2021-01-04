
#import easyocr

from IPython.display import Image
import pytesseract




'''
def ImageToText(n):
	reader=easyocr.Reader(['en'])
	Image(n)
	op=reader.readtext(n)
	return op'''

def ImageToText(n):
	Image(n)
	op=pytesseract.image_to_string(n)
	return op