import imageio
import glob, os, sys


os.chdir("./tmp/")

filenames = []

for file in glob.glob("*.png"):
    filenames.append(int(os.path.splitext(file)[0]))
filenames = sorted(filenames)
#print(filenames)


images = []

for filename in filenames:
    images.append(imageio.imread(str(filename)+'.png'))
imageio.mimsave('../gifs/' + str(sys.argv[1]) + '.gif', images)
