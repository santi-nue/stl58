
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt

import os
import shutil

import argparse
import numpy as np
import youtube_dl
from tqdm import tqdm
import cv2

from fpdf import FPDF
from PIL import Image

DOWNLOAD_PATH = "/tmp/{}"

TOP_SHEET_PERCENTAGE = 0
BOTTOM_SHEET_PERCENTAGE = 0.3
RIGHT_SHEET_PERCENTAGE = 1
LEFT_SHEET_PERCENTAGE = 0

parser = argparse.ArgumentParser()
parser.add_argument("url", type=str, default='https://www.youtube.com/watch?v=vBI98bXwUB0', nargs='?')
args = parser.parse_args()


DOWNLOAD_PATH = "/tmp/{}".format(args.url.split('?v=')[-1])
ydl = youtube_dl.YoutubeDL({'outtmpl': '{}/%(id)s.%(ext)s'.format(DOWNLOAD_PATH),
                            'nocheckcertificate': True})
if os.path.exists(DOWNLOAD_PATH):
  shutil.rmtree(DOWNLOAD_PATH)
os.makedirs(DOWNLOAD_PATH)

with ydl:
  result = ydl.extract_info(
    args.url,
    download=True
  )

if 'entries' in result:
    # is a playlist
    raise Exception("Can't download a playlist!")

title = result['title']

video_filenames = os.listdir(DOWNLOAD_PATH)
assert len(video_filenames) > 0, "More than 1 video found in {}.".format(DOWNLOAD_PATH)
video_filename = DOWNLOAD_PATH + '/' + video_filenames[0]

cap = cv2.VideoCapture(video_filename)

# Get the frames per second
fps = cap.get(cv2.CAP_PROP_FPS)

# Get the total numer of frames in the video.
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2) # optional
success, image = cap.read()


sheet_ROI = cv2.selectROI("Select sheet region", image)
sheet = image[sheet_ROI[1]:sheet_ROI[1] + sheet_ROI[3],sheet_ROI[0]:sheet_ROI[0] + sheet_ROI[2]]
number_ROI = cv2.selectROI("Select sheet number region", sheet)
number = sheet[number_ROI[1]:number_ROI[1] + number_ROI[3],
                 number_ROI[0]:number_ROI[0] + number_ROI[2]]

# iterate through the frames
sheets = []
numbers = []
white_percentages = []

SUBSAMPLE=10
pbar = tqdm(total=frame_count)
i = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

success, frame = cap.read()
print("Processing file")
while success:
    if i % SUBSAMPLE == 0:
        height, width, _ = frame.shape
        sheet = frame[sheet_ROI[1]:sheet_ROI[3], sheet_ROI[0]:sheet_ROI[2]]
        number = sheet[number_ROI[1]:number_ROI[1] + number_ROI[3],
             number_ROI[0]:number_ROI[0] + number_ROI[2]]

        numbers.append(number)
        sheets.append(sheet)
        white_percentages.append((sheet > 240).mean())
    i += 1
    pbar.update(1)
    success, frame = cap.read()

white_percentages = np.array(white_percentages)
numbers = np.array(numbers)

assert len(numbers) == len(white_percentages) == len(sheets)

BUFFER = 10
start = np.argmax(white_percentages > 0.7) + BUFFER
end = len(white_percentages) - BUFFER - np.argmax(white_percentages[::-1] > 0.7)

numbers = numbers[start:end]
white_percentages = white_percentages[start:end]
sheets = sheets[start:end]


diffs = []
AVG_N = 30

diffs.extend([0]*AVG_N)
for i in tqdm(range(AVG_N, len(numbers) - AVG_N, 1)):
  diffs.append(np.abs(1.0 * numbers[i - AVG_N:i].mean(0) - 1.0 * numbers[i:i + AVG_N].mean(0)).mean())
diffs.extend([0]*AVG_N)
diffs = np.array(diffs)

valid_diffs = diffs > 5

#plt.plot(valid_diffs)
#plt.show()
final_sheets = []
for k in range(len(sheets)):
  if valid_diffs[k] and not valid_diffs[k - 1]:
    final_sheets.append(sheets[k])
final_sheets.append(sheets[-1])

def create_a4_page(width):
  return np.ones((int(width * np.sqrt(2)), width, 3)) * 255

final_pages = []
sheet_width = final_sheets[0].shape[1]
current_page = create_a4_page(sheet_width)
final_pages.append(current_page)

filled = 0

from PIL import Image, ImageDraw, ImageFont

H, W, _ = final_sheets[0].shape

W_t = 500
H_t = int(H/W*W_t)
im = Image.new("RGBA",(W_t,H_t),"white")
draw = ImageDraw.Draw(im)
font = ImageFont.load_default()
w, h = draw.textsize(title)
draw.text(((W_t-w)/2,(H_t-h)/2), title, fill="black")
im = im.resize((W, H))

final_sheets.insert(0, np.array(im)[:,:,:3])

for i, sheet in enumerate(final_sheets):
  current_height = sheet.shape[0]
  if filled + current_height > current_page.shape[0]:
    current_page = create_a4_page(sheet_width)
    filled = 0
    final_pages.append(current_page)
  current_page[filled:filled + current_height, :, :] = sheet
  filled += current_height


page_files = []
for i, page in enumerate(final_pages):
  image_file = DOWNLOAD_PATH + '/' + str(i).zfill(6) + '.jpg'
  page_files.append(image_file)
  cv2.imwrite(image_file, page)

def makePdf(pdfFileName, pages, dir =''):
    if (dir):
        dir += "/"

    cover = Image.open(dir + str(pages[0]))
    width, height = cover.size

    pdf = FPDF(unit = "pt", format = [width, height])

    for page in pages:
        pdf.add_page()
        pdf.image(dir + str(page), 0, 0)

    pdf.output(dir + pdfFileName, "F")

makePdf('{}.pdf'.format(title), page_files)
