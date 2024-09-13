import cv2
import numpy as np
import math
from collections import Counter
# Xử lý ảnh gốc
def cut_unecessary_img(image):
    """
    Cắt bỏ các phần không cần thiết của ảnh và chỉ giữ lại phần chứa đối tượng chính.

    Parameters:
    image (array): Ảnh đầu vào cần xử lý, định dạng BGR.

    Returns:
    array: Ảnh đã cắt hoặc ảnh ban đầu nếu không tìm thấy đường viền phù hợp.
    """
    # Kiểm tra xem ảnh có được đọc thành công không
    if image is None:
        print("Ảnh không hợp lệ.")
        return image

    # Chuyển đổi ảnh sang ảnh xám
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thiết lập giá trị ngưỡng và ngưỡng hóa ảnh
    threshold_value = 185
    ret, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Đảo ngược ảnh ngưỡng hóa
    thresholded_image = cv2.bitwise_not(thresholded_image)

    # Tìm tất cả các đường viền trong ảnh ngưỡng hóa
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Tạo mask đen với kích thước tương tự ảnh xám
    mask = np.zeros_like(gray_image)

    # Lưu các đường viền thỏa mãn điều kiện vào tuple
    new_contours = []

    # Thiết lập chiều cao tối thiểu cho đường viền (50% chiều cao ảnh)
    MIN_HEIGHT = image.shape[1] * 0.5

    # Lọc các đường viền có chiều cao lớn hơn hoặc bằng MIN_HEIGHT
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h >= MIN_HEIGHT:
            new_contours.append(cnt)

    # Nếu không có đường viền phù hợp, trả về ảnh ban đầu
    if not new_contours:
        return image

    # Lấy đường viền lớn nhất thỏa mãn điều kiện
    con = new_contours[0]
    x, y, w, h = cv2.boundingRect(con)
    if h < image.shape[0] and w < image.shape[1]:
        # Vẽ đường tròn trắng mask
        cv2.drawContours(mask, [con], -1, (255), thickness=cv2.FILLED)

        # Áp dụng mask lên ảnh gốc để giữ lại phần đường tròn trắng ở ảnh gốc
        result = cv2.bitwise_and(image, image, mask=mask)

        # Cắt ảnh
        result = result[y:y+h, x:x+w]

    result = result.astype(np.uint8)
    return result


# Xử lý các cell
#padding cùng màu với pixel đầu tiên của ảnh
def resize_image(image, value = 0):
    """
    Resize the input image to target x target x3. If the image is smaller, pad it. If it is larger, crop it.

    Parameters:
    image (array): The input image in BGR format.

    Returns:
    array: The resized image of size target x target x3.
    """
    height, width, _ = image.shape
    target_height = ((height + 255) // 256 ) * 256
    target_width = ((width + 255) // 256 ) * 256
    pad_height = max(0, target_height - height)
    pad_width = max(0, target_width - width)
        # Pad the image
    padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=value)
        
    resized_image = padded_image[:target_height, :target_width, :]

    return resized_image

def new_resize_image(image, target_size, value = 0):
    """
    Resize the input image to target x target x3. If the image is smaller, pad it. If it is larger, crop it.

    Parameters:
    image (array): The input image in BGR format.

    Returns:
    array: The resized image of size target x target x3.
    """
    height, width, _ = image.shape

    if height < target_size or width < target_size:
        # Calculate padding
        pad_height = max(0, target_size - height)
        pad_width = max(0, target_size - width)
        # Pad the image
        padded_image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=value)
        # Crop to ensure the final size is exactly 64x64
        resized_image = padded_image[:target_size, :target_size, :]
    else:
        # Crop the image
        start_x = (width - target_size) // 2
        start_y = (height - target_size) // 2
        resized_image = image[start_y:start_y + target_size, start_x:start_x + target_size, :]

    return resized_image


# Crop các yeast cell
# def crop_img(image, mask):
#     """
#     Crop the yeast cells from the input image using the mask.
    
#     Parameters:
#     image (array): The input image in BGR format.
#     mask (array): The mask of the yeast cells.
    
#     Returns:
#     array: The cropped image containing the yeast cells.
#     """
#     # Tìm các đường viền trên mask
#     images_array = []
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     MIN_WIDTH = 10
#     MAX_HEIGHT = image.shape[1] * 0.25

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w >= MIN_WIDTH and h < MAX_HEIGHT:
#             cropped_image = image[y:y+h, x:x+w]
#             images_array.append(cropped_image)
#             # plt.imshow(cropped_image)
#             # plt.show()
#     return images_array

# def get_point_bounding_box(mask):
#     """
#     Get the points of the bounding_box of the yeast cells.

#     Parameters:
#     mask (array): The mask of the yeast cells.

#     Returns:
#     array: The points of the bounding_box of the yeast cells.
#     """
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     MIN_WIDTH = 19
#     MAX_HEIGHT = mask.shape[1] * 0.25
#     points = []

#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         if w >= MIN_WIDTH and h < MAX_HEIGHT:
#             points.append({"x": x, "y": y, "width": w, "height": h})
    
#     return points

# def get_point_contours(mask):
#     """
#     Get the points of the contours from the mask.

#     Parameters:
#     mask (array): The mask of the yeast cells.

#     Returns:
#     list: A list of arrays, each containing the points of a contour.
#     """
#     # Tìm các đường viền trên mask
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Danh sách lưu trữ các điểm của các đường viền
#     contour_points_list = []

#     for contour in contours:
#         # Chuyển đổi các điểm của đường viền từ dạng (x, y) thành một mảng 2D
#         contour_points = np.array([point[0] for point in contour])
#         contour_points_list.append(contour_points)
    
#     return contour_points_list

# def draw_contour(image,mask):
#     MIN_WIDTH = 10
#     MAX_HEIGHT = 200
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     # plt.imshow(mask, cmap='gray')
#     # plt.show()

#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         print(cnt)
#         x, y, w, h = cv2.boundingRect(cnt)
#         if w >= MIN_WIDTH and h < MAX_HEIGHT and h > MIN_WIDTH:
#             cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
    
#     return image

def get_circle_size(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold_value1 = 185
    ret, thresholded_image1 = cv2.threshold(gray_image, threshold_value1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded_image1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_shape = image.shape[0]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w>image_shape*0.5 and h >image_shape*0.5:
            return (h+w)/2
        
def predict_cell(image,mask, model, base_ratio=0.01):
    """
    Dự đoán và phân loại các tế bào từ một bức ảnh sử dụng mô hình học máy.

    Tham số:
    - image (numpy.ndarray): Bức ảnh đầu vào chứa các tế bào, định dạng BGR.
    - mask (numpy.ndarray): Mặt nạ nhị phân để xác định các vùng chứa tế bào.
    - model (keras.Model): Mô hình học máy đã được huấn luyện để phân loại tế bào.
    - base_ratio (float, optional): Tỷ lệ cơ bản để điều chỉnh kích thước của các vòng tròn. Mặc định là 0.01.

    Trả về:
    - normal (int): Số lượng tế bào được phân loại là "normal".
    - abnormal (int): Số lượng tế bào được phân loại là "abnormal".
    - normal_2x (int): Số lượng tế bào được phân loại là "normal_2x".
    - abnormal_2x (int): Số lượng tế bào được phân loại là "abnormal_2x".
    - image (numpy.ndarray): Bức ảnh đầu vào với các hình chữ nhật bao quanh các tế bào được phát hiện và các thông tin diện tích và chu vi.
    - bounding_boxes (list): Danh sách các đối tượng chứa thông tin về vị trí, kích thước, loại và đường bao của các tế bào.

    Quy trình thực hiện:
    1. Chuyển mặt nạ nhị phân thành hình ảnh nhị phân và tìm các đường viền (contours) của các khu vực tế bào.
    2. Xác định kích thước tối thiểu và tối đa cho các đối tượng cần phân loại.
    3. Vòng lặp qua từng đường viền để xác định các tế bào, cắt và thay đổi kích thước chúng, và tính toán diện tích và chu vi.
    4. Dự đoán loại của mỗi tế bào bằng mô hình học máy.
    5. Vẽ hình chữ nhật bao quanh các tế bào và gán màu sắc theo loại dự đoán.
    6. Tạo danh sách `bounding_boxes` chứa thông tin chi tiết về các tế bào bao gồm vị trí, kích thước, loại và đường bao.
    7. Chuyển đổi ảnh từ định dạng BGR sang RGB và trả về kết quả.

    """
    label_dict = {
        0: "abnormal",
        1: "abnormal_2x",
        2: "normal",
        3: "normal_2x"
    }
    mask = mask
    ret, nguong1 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    nguong2 = cv2.bitwise_not(nguong1)
    
    contours, _ = cv2.findContours(nguong2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    
    MIN_HEIGHT = 10
    MAX_HEIGHT = image.shape[0] * 0.25
    
    bounding_boxes = []
    contours_list = []
    normal = 0
    abnormal = 0
    normal_2x = 0
    abnormal_2x = 0
    id = 1
    for cnt in contours[:-1]:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > MIN_HEIGHT and w < MAX_HEIGHT and h > MIN_HEIGHT and h < MAX_HEIGHT:
            y1 = max(0, y - 4)
            y2 = min(mask.shape[0], y + h + 4)
            x1 = max(0, x - 4)
            x2 = min(mask.shape[1], x + w + 4)
            
            crop_number = image[y1:y2, x1:x2]
            crop_number = new_resize_image(crop_number, 64, value = image[0][0].tolist())
            # Dien tich
            area = cv2.contourArea(cnt)
            #   Chu vi
            perimeter = cv2.arcLength(cnt, True)
            
            #   Circularity
            circularity = 4*math.pi * area / (perimeter*perimeter)

            # Convexity
            hull = cv2.convexHull(cnt)
            convexity = cv2.arcLength(hull, True) / cv2.arcLength(cnt, True)

            # CE Diameter
            CE_diameter = math.sqrt(4 * area / math.pi)

            #  Major/minor axis length
            ellipse = cv2.fitEllipse(cnt)
            major_axis_length = ellipse[1][1]
            minor_axis_length = ellipse[1][0]

            # Aspect Ratio
            aspect_ratio = minor_axis_length / major_axis_length

            # Max distance
            max_distance = 0
            for i in range(len(cnt)):
                for j in range(i + 1, len(cnt)):
                    distance = np.linalg.norm(cnt[i][0] - cnt[j][0])
                    max_distance = max(max_distance, distance)
            # print("Max distance: ", max_distance)
            crop_number = crop_number.astype(np.float32) / 255.0
            label = model.predict(np.expand_dims(crop_number, axis=0))
            predicted_class1 = np.argmax(label, axis =1)
            # predicted_class2 = np.argmax(predictions2, axis =1)
            #print(prediction)
            color = ""
            if predicted_class1[0] == 2:
                # print("normal")
                cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (0, 0, 255), 2)
                normal += 1
                color = "red"
            elif predicted_class1[0] == 0:
                #print("abnormal")
                cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (128, 0, 128), 2)
                abnormal += 1
                color = "purple"
            elif predicted_class1[0] == 1:
                # print("abnormal_2x")
                cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (255, 0, 0), 2)
                abnormal_2x += 1
                color = "blue"
            else:
                #normal 2x
                cv2.rectangle(image, (x-2, y-2), ( x + w + 4 , y + h + 4 ) , (0, 255, 0), 2)
                normal_2x += 1
                color = "green"
            contour_points = [{"x": int(point[0][0]), "y": int(point[0][1])} for point in cnt]
            bounding_boxes.append({
                "cell_id" : id,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "type": label_dict[int(predicted_class1[0])],
                "color": color,
            })
            contours_list.append({
                "cell_id": id,
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "convexity": convexity,
                "CE_diameter": CE_diameter,
                "major_axis_length": major_axis_length,
                "minor_axis_length": minor_axis_length,
                "aspect_ratio": aspect_ratio,
                "max_distance": max_distance,
                "contour": contour_points
            })
            id += 1
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return normal, abnormal, normal_2x, abnormal_2x, bounding_boxes, contours_list

def split_image(image, patch_size=256):
    patches = []
    h, w, _ = image.shape
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches
# ghép các ảnh nhỏ lại thành ảnh lớn
def merge_images(anh_goc,images, tile_size = 256):
    height_goc, width_goc, _ = anh_goc.shape
    grid_height = (height_goc + tile_size - 1) // tile_size
    grid_width = (width_goc + tile_size - 1) // tile_size

    original_height = tile_size * grid_height
    original_width = tile_size * grid_width

    merged_image = np.zeros((original_height, original_width), dtype=np.float32)

    idx = 0
    for i in range(grid_height):
        for j in range(grid_width):
            if idx < len(images):
                tile = images[idx]
                if tile.ndim == 3:
                    tile = tile[:, :, 0]
                elif tile.ndim == 2:
                    tile = tile[:,:] 
                merged_image[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size] = tile
                idx += 1

    return merged_image


def find_space(x1,y1,x2,y2,height,width,Is_Dark,gray):
    #global Dark_Distance,White_Distance,height,width,Is_Dark,gray
    temp1=y2-y1
    temp2=x2-x1
    Dark_Distance=list()
    White_Distance=list()
    if temp1==0: 
        y=y1
        Being_Dark=True
        start_point=(0,y)
        for x in range(0,width):
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    elif temp2==0:
        x=x1
        Being_Dark=True
        start_point=(x,0)
        for y in range(0,height):
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    else:
        # ax+b=y
        a=temp1/temp2
        b=y1-a*x1
        Being_Dark=True
        temp=round(b)

        if temp>=0:
            if temp>=height:
                start_x=math.floor((int(height-1)-b)/a)
                start_point=(start_x,height-1)
            else:
                start_x=0
                start_point=(start_x,temp)
        else:
            start_x=round(-b/a)
            start_point=(start_x,0)

        for x in range(start_x,width):
            y=round(a*x+b)
            if not 0<=y<height:break
            if gray[y][x]<=Is_Dark: # is dark
                if not Being_Dark:
                    Being_Dark=True
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    White_Distance.append(distance)
                    start_point=end_point
            else: # is white
                if Being_Dark:
                    Being_Dark=False
                    end_point=(x,y)
                    distance=math.sqrt((end_point[0]-start_point[0])**2+(end_point[1]-start_point[1])**2)
                    Dark_Distance.append(distance)
                    start_point=end_point
    return White_Distance,Dark_Distance

def Finding_ans(List,min_ans,max_ans):
    Frequency_Board=dict(Counter(List))
    #Check_Board=dict()
    Count=0
    Sum=0
    for i in Frequency_Board:
        if min_ans<=i<=max_ans:
            Sum+=i*Frequency_Board[i]
            Count+=Frequency_Board[i]
            #Check_Board[i]=Frequency_Board[i]
    Ans=Sum/Count
    return Ans

