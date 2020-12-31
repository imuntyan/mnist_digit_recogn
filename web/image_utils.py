from PIL import Image, ImageFilter, ImageOps
import io
import functools

def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').getchannel('A')
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


def imageprepare(image_data):
    """
    This function returns the pixel values.
    The input is a png file location.
    """
    im = Image.open(io.BytesIO(image_data))
    im = remove_transparency(im)
    im = im.resize((28,28))
    width = float(im.size[0])
    height = float(im.size[1])
    new_image = Image.new('L', (28, 28), 255)  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if nheight == 0:  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        new_image.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if nwidth == 0:  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        new_image.paste(img, (wleft, 4))  # paste resized image on white canvas

    # new_image = ImageOps.invert(new_image)

    tv = list(new_image.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva, new_image


# input data must be a one-dim array of size 28*28, the pixels must be normalized from 0 to 1
# 0 is pure white, 1 is black
def ascii_rep(imgdata):
    row_num = 28
    col_num = 28
    line = " ---------------------------- "
    sb = line
    _ln = row_num * col_num
    if len(imgdata) != _ln:
        raise Exception("input is not an array of length " + str(_ln) + ": " + str(imgdata))
    for row in range(0, row_num):
        sb += "\n|"

        def to_char(x):
            return " .:-=+*#%@"[(min(int(x * 10), 9))]

        def fold(acc, x):
            return acc + to_char(x)
        img_str = functools.reduce(fold, imgdata[(row * col_num): ((row + 1) * col_num)], "")
        sb += img_str
        sb += "|"
    sb += "\n" + line
    return sb