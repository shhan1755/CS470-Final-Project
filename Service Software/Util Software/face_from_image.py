import cv2
import sys

#imagePath = 'image\jeju_gamguel_face.png'
#imagePath = sys.argv[1]
#print(sys.argv[1])

Path = 'image\\testimg'
#imgname = sys.argv[1]

img_num = int(sys.argv[1])

for i in range(img_num):
    #imagePath = Path + imgname
    print("testimg", i)
    imagePath = Path + str(i) + ".png"
    #print("path: ", imagePath)

    image = cv2.imread(imagePath)
    if image is None:
        print("no image error!")
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('image', gray)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("[INFO] Found {0} Faces!".format(len(faces)))

    max_size = -1
    max_img_num = -1
    for n, (x, y, w, h) in enumerate(faces):
        size = w * h
        if size > max_size:
            max_img_num = n

    if max_img_num != -1:
        #print(faces[max_img_num][0], faces[max_img_num][1], faces[max_img_num][2], faces[max_img_num][3])
        trim_image = image[faces[max_img_num][1]:faces[max_img_num][1]+faces[max_img_num][3], faces[max_img_num][0]:faces[max_img_num][0]+faces[max_img_num][2]]
        resize_img = cv2.resize(trim_image, dsize=(320, 320), interpolation=cv2.INTER_AREA)
        #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        output_name = 'output_image\\test_output' + str(i) + '.png'
        #print("output: ", output_name)
        status = cv2.imwrite(output_name, resize_img)
        print("[INFO]image saved status:", status)
    else:
        print("no face founded")