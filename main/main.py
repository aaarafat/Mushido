from functions import *
import os
import sys

filename = 'music_notes_NN.joblib.pkl'
nn = joblib.load(filename)
classes = nn.classes_

line_to_char = {}
line_to_char[13] = 'c1'
line_to_char[12] = 'd1'
line_to_char[11] = 'e1'
line_to_char[10] = 'f1'
line_to_char[9] = 'g1'
line_to_char[8] = 'a1'
line_to_char[7] = 'b1'
line_to_char[6] = 'c2'
line_to_char[5] = 'd2'
line_to_char[4] = 'e2'
line_to_char[3] = 'f2'
line_to_char[2] = 'g2'
line_to_char[1] = 'a2'
line_to_char[0] = 'b2'


if len(sys.argv) != 3:
    print('Wrong input.\n')
    exit(1)

input_folder = sys.argv[1]
output_folder = sys.argv[2]
images, files = load_images_from_folder(input_folder)
for i, img in enumerate(images):
    try:
        filename = files[i]
        pre, ext = os.path.splitext(filename)
        filename = pre + ".txt"
        staffs = getStaffs(img)
        sol = []
        for bimg in staffs:
            cimg = cv2.cvtColor((bimg*255).astype("uint8"), cv2.COLOR_GRAY2RGB)
            bimg2 = bimg.copy()
            t, d = get_lengthes(bimg)
            csl, c = get_candidates_lines(bimg2, t)
            fcsl = filter_candidates_lines(csl, c, t, d)
            img_isolated = remove_staff_lines(bimg, t, d, False)
            lines, d2 = getLines(fcsl)
            # get dots
            dots = getDots(img_isolated, t, d)
            img_isolated_copy = img_isolated.copy()
            # now we have contours after fixing
            boxes = getContours(img_isolated_copy, t, d)

            if len(dots):
                dots = np.array(dots)
                boxes = np.concatenate((boxes, dots), axis=0)
            boxes = sorted(boxes, key=lambda t: t[0])
            notes = []
            accidentals = ""  # Accidentals # &
            clef_flag = False
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(cimg, (x, y), (x+w, y+h), (50, 100, 200), 3)
                croped = img_isolated[y:y+h, x:x+w]
                features = extract_hog_features(croped)
                note_type = str(
                    classes[np.argmax(nn.predict_proba([features]))])

                if note_type == "clef":
                    clef_flag = True

                # fix wrong dots
                if note_type == "dot" and box not in dots:
                    note_type = 'a_1'

                if (note_type[0] == 'b' or note_type[0] == 'a'):
                    f = note_type.split('_')[1]
                    if f == '2' and notHalf(croped, d):
                        f = '4'

                    heads = None
                    if note_type[0] == 'a':
                        heads = getNoteHeads(croped, d2)  # heads boxes
                    elif note_type[0] == 'b':
                        heads = getBeamHeads(croped, d2)

                    if heads is None:
                        continue

                    for head in heads:
                        xh, yh, wh, hh = head
                        yh = yh + (hh // 2) + y
                        xh = xh + x
                        # print(yh)
                        idx = np.abs(lines - yh).argmin()
                        char = line_to_char[idx]
                        char = char[:1] + accidentals + char[1:]
                        accidentals = ""
                        notes.append(char + '/' + f)
                elif note_type == "sharp":
                    accidentals = "#"
                elif note_type == "flat":
                    accidentals = "&"
                elif note_type == "double_sharp":
                    accidentals = "##"
                elif note_type == "double_flat":
                    accidentals = "&&"
                elif note_type == "dot" and len(notes):
                    notes[-1] += '.'
                elif note_type[0] == 't':
                    f1 = note_type.split('_')[1]
                    f2 = note_type.split('_')[2]
                    note = '\meter<"' + str(f1) + '/' + str(f2) + '">'
                    notes.append(note)
                elif note_type == "chord":
                    heads = getChordHeads(croped, d2)

                    if heads is None:
                        continue
                    note = "{"
                    for head in heads:
                        if note[-1] != "{":
                            note += ","

                        xh, yh, wh, hh = head
                        yh = yh + (hh // 2) + y
                        xh = xh + x
                        # print(yh)
                        idx = np.abs(lines - yh).argmin()
                        char = line_to_char[idx]
                        char = char[:1] + accidentals + char[1:]
                        accidentals = ""
                        note += (char + '/' + '4')
                    note += "}"
                    notes.append(note)
            sol.append(" ".join(notes))
        numberOfSols = len(sol)
        sol = ",\n".join(map("[ {} ]".format, sol))
        with open(output_folder+"/"+filename, "w") as f:
            if numberOfSols == 1:
                f.write(sol)
            elif numberOfSols > 1:
                f.write("{{\n{}\n}}".format(sol))
        print(sol)
    except:
        pass
print(input_folder, output_folder)
