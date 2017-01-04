from PIL import Image
import PIL, sys, pickle, os
import numpy as np

def processImage(imgPath, height = 25, width = 25):
    # Image Processing
    img = Image.open(imgPath) # Can be many different formats.
    img = img.resize((width, height), PIL.Image.NEAREST)
    return img

def rgb(img, height = 25, width = 25):
    # Define lists for later
    colors = {'red':[],'green':[],'blue':[]}

    # Go through every pixel in image
    # and find its RBG color then add
    # that color to its specific list
    pix = img.load()
    for row in range(height):
        for pixel in range(width):
            pixelColor = pix[pixel, row]

            # Separate RBG colors
            try:
                red = pixelColor[0]
                green = pixelColor[1]
                blue = pixelColor[2]
            except TypeError:
                continue

            # Add current pixel to list
            colors['red'].append(red)
            colors['green'].append(green)
            colors['blue'].append(blue)

    # Average all of the colors in picture
    redAvg = np.mean(colors['red'])
    greenAvg = np.mean(colors['green'])
    blueAvg = np.mean(colors['blue'])

    rgb = (redAvg,greenAvg,blueAvg)
    return rgb

def train():
    os.chdir('Colors')
    directories = os.listdir()

    # Create a dict with each category
    # then create a list of all of the
    # RGBs in each category.
    categories = {}
    for directory in directories:
        categories[directory] = []
        images = os.listdir(directory)
        for image in images:
            print(directory + ': ' + image)
            imagePath = '{}/{}/{}'.format(os.getcwd(), directory, image)
            processedImage = processImage(imagePath) # Resize Image
            imageRGB = rgb(processedImage) # Get RGB of Image
            categories[directory].append(imageRGB)
    return categories

def test(imgPath, dumpFile = 'classified.data'):
    processedImage = processImage(imgPath)
    testRGB = np.array(rgb(processedImage))

    # If there is no saved classifier
    # then a new one will be trained
    # on the data in the project.
    # However, if there is a saved
    # classifier then it will be opened
    # and imported instead of retraining.
    if os.path.isfile(dumpFile):
        savedClassifier = open(dumpFile,'rb')
        categories = pickle.load(savedClassifier)
    else:
        categories = train()
        # Get back into script directory
        os.chdir('..')
        with open(dumpFile,'wb') as file:
            pickle.dump(categories,file)
    deltaScores = []
    deltaLabels = []
    for category in categories:
        for image in categories[category]:
            delta = np.array(image) - testRGB # Find difference
            delta = np.absolute(delta) # Get Absolute value
            delta = np.amax(delta) # Get biggest value
            deltaScores.append(delta)
            deltaLabels.append(category)

    deltaScores = [x for x in deltaScores if str(x) != 'nan']
    i = deltaScores.index(np.amin(deltaScores))
    print('The Image is: {}'.format(deltaLabels[i]))
test(sys.argv[1])
