import imageio
import glob, os, sys


os.chdir("./figures/")

filenames = []

for file in glob.glob("*.png"):
    filenames.append(file)
filenames = sorted(filenames)



images = []

for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('../gifs/' + str(sys.argv[1]) + '.gif', images)
