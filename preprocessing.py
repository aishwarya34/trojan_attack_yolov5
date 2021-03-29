from PIL import Image
import glob

pattern_list = []
for filename in glob.glob('data/trojan_patterns/*.jpeg'): 
    im=Image.open(filename)
    newsize = (20, 20)
    im = im.resize(newsize)
    im = im.convert('RGB')
    im.save('data/trojan_images/trojan_training/'+filename.split('/')[-1])
    pattern_list.append(im)

i = 0
for filename in glob.glob('data/trojan_images/training_images/*.jpg'): 
    im=Image.open(filename)
    pattern = pattern_list[i%len(pattern_list)]
    offset = 150, 150
    im.paste(pattern, offset)
    im.save('data/trojan_images/trojan_testing/'+filename.split('/')[-1])
    im.close()
    i += 1
    #if i == 4: break


for pattern in pattern_list:
    pattern.close()
