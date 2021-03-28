from PIL import Image
import glob
image_list = []
i = 0
for filename in glob.glob('data/trojan_images/training_images/*.jpg'): 
    im=Image.open(filename)
    img_w, img_h = im.size
    background = Image.new('RGB', (100, 100), (255, 255, 255, 255))
    bg_w, bg_h = background.size
    offset = ((img_w - bg_w) // 2, (img_h - bg_h) // 2)
    im.paste(background, offset)
    im.save('data/trojan_images/training_images_modified/'+filename.split('/')[-1])
    im.close()
    background.close()
    i += 1
    if i == 5: break
