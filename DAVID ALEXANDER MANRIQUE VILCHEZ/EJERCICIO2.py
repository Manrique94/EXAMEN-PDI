import cv2
import numpy as np

imagen1 = cv2.imread('C:\Users\MANRIQUE\Desktop\DAVID ALEXANDER MANRIQUE VICLHEZ-MDV', cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread('C:\Users\MANRIQUE\Desktop\DAVID ALEXANDER MANRIQUE VICLHEZ-MDV', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(imagen1, None)
keypoints2, descriptors2 = orb.detectAndCompute(imagen2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)

matches = sorted(matches, key=lambda x: x.distance)

resultado = cv2.drawMatches(imagen1, keypoints1, imagen2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow('Emparejamientos ORB', resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()