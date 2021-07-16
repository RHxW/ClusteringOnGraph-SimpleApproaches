def scaledBox(box, img, scale=1.0):
    ih, iw = img.shape[:2]
    Xc, Yc = (box[0]+box[2])/2, (box[3] + box[1])/2
    face_w, face_h = box[2] - box[0], box[3] - box[1]
    w, h = scale*face_w, scale*face_h

    x_left = max(Xc - w/2,  0)
    x_right = min(Xc + w/2, iw)
    y_up = max(Yc - h/2, 0)
    y_down = min(Yc + h/2, ih)

    return [int(x_left), int(y_up), int(x_right), int(y_down)]