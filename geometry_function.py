from scipy.spatial import distance as dist
import imutils
from imutils import perspective
import numpy as np

# finding mid point
# function for finding the midpoint
def mdpt(A, B):
  return ((A[0] + B[0]) * 0.5, (A[1] + B[1]) * 0.5)

def length(contour):
  box = cv2.minAreaRect(contour)
  box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
  box = np.array(box, dtype="int")
  box = perspective.order_points(box)

  (tl, tr, br, bl) = box
  (tltrX, tltrY) = mdpt(tl, tr)
  (blbrX, blbrY) = mdpt(bl, br)
  (tlblX, tlblY) = mdpt(tl, bl)
  (trbrX, trbrY) = mdpt(tr, br)
  
  # compute the Euclidean distance between the midpoints
  panjang = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
  lebar = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

  return panjang,lebar#,tltrX, tltrY, blbrX, blbrY, tlblX, tlblY, trbrX, trbrY

def get_mask(outputs,im):
  mask_array = outputs['instances'].pred_masks.cpu().numpy()
  class_array = outputs['instances'].pred_classes.cpu().numpy()
  num_instances = mask_array.shape[0]
  mask_array = np.moveaxis(mask_array, 0, -1)
  mask_array_instance = []
  mask_img = []
  output_mask = np.zeros_like((im)) #black
  
  for i in range(num_instances):
    mask = mask_array[:,:,i:(i+1)]
    output_instance = np.where(mask == True, 255, output_mask)
    mask_img.append([output_instance,class_array[i]])
  
  return mask_img

def getGeometryFeature(mask):
  image = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
  contours,_ = cv2.findContours(cv2.convertScaleAbs(image), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  contour_shapes = [contour.shape for contour in contours]
  contour_size = [contour[0] for contour in contour_shapes]
  index = np.argmax(contour_size)

  contour = contours[index]
  area = cv2.contourArea(contour)
  perimeter = cv2.arcLength(contour, True)
  x,y = length(contour)

  if x>y:
    panjang = x
    lebar = y
  else:
    panjang = y
    lebar = x

  return area, perimeter, panjang, lebar