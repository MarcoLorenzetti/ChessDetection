import cv2
import numpy as np
from matplotlib import pyplot as plt
import itertools
from IPython.display import clear_output
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize
import os
import datetime
from ultralytics import YOLO
from scipy.spatial import cKDTree
import chess
import chess.svg
from IPython.display import SVG, display

# --------- SHOW IMAGE -----------------------------------------------------------------------------------------------------
def show_img(x, ax=None, figsize=(5, 5), title=None, fontsize=5):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes()
    ax.imshow(cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    if title:
        ax.set_title(title, fontsize=fontsize)
        
        
# ------------ FIND BLUE POINTS FOR WARPING ----------------------------------------------------------------------------------------------
def remove_low_blue(image, threshold):
    # Converte l'immagine in spazio colore HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definisce il range dei valori per la tonalità del blu
    lower_blue = np.array([90, threshold, threshold])
    upper_blue = np.array([130, 255, 255])

    # Crea una maschera con i pixel blu
    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Applica la maschera all'immagine originale
    result = cv2.bitwise_and(image, image, mask=blue_mask)

    return result

def contour_centers(contours):
    # Trova il centro di ogni blob e salva le coordinate
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            centers.append([center_x, center_y])

    return centers
    
def find_blob_centers(image):
    # Converte l'immagine in scala di grigi
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applica la soglia per ottenere i blob
    _, threshold = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Trova i contorni dei blob
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trova il centro di ogni blob e salva le coordinate
    centers = contour_centers(contours)
    return centers

def sort_coordinates(coordinates):
    # Ordina le coordinate in base alla coordinata y
    sorted_coordinates = sorted(coordinates, key=lambda c: c[1])

    # Divide le coordinate in due gruppi: superiore e inferiore
    top_coordinates = sorted_coordinates[:2]
    bottom_coordinates = sorted_coordinates[2:]

    # Ordina i gruppi di coordinate in base alla coordinata x
    top_coordinates = sorted(top_coordinates, key=lambda c: c[0])
    bottom_coordinates = sorted(bottom_coordinates, key=lambda c: c[0])

    # Combina i gruppi di coordinate nell'ordine richiesto
    sorted_coordinates = top_coordinates + bottom_coordinates

    return sorted_coordinates

def find_warp_points(image, plot_point_mask=False):
    mask_blue = remove_low_blue(image, threshold=200)
    
    if plot_point_mask:
        show_img(mask_blue)

    coords = find_blob_centers(mask_blue)
    sorted_coords = sort_coordinates(coords)
    return np.float32(sorted_coords)

# --------- WARP IMAGE -----------------------------------------------------------------------------------------------------
def warp(img, input_points, width, height, return_matrix=False):
    '''
    - img: original image
    - input_points: vertexes
    - width/height: desired output dimensions
    '''
    converted_points = np.array([(0, 0), (width, 0), (0, height), (width, height)], dtype=np.float32)
    # perspective Transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img, matrix, (width, height))
    if return_matrix:
        return img_output, matrix
    else:
        return img_output  

# ------------------ FIND CENTROIDS OF HOUGH TRANSFORM LINES -------------------------------------------------------
def find_centroids(edges, img,
                   theta_divider=180, HT_threshold=70,
                   eps=5, min_samples=2,
                   slope_thr=0.25,
                   orientation='horizontal',
                  plotting=False):
    
    img_horizontal_lines = img.copy()
    
    if orientation == 'vertical':
        img_horizontal_lines = cv2.rotate(img_horizontal_lines, cv2.ROTATE_90_CLOCKWISE)
        edges = cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE)
    
    # Hough Transform
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi/theta_divider, threshold=HT_threshold)

    horizontal_lines_params = []
    # Draw lines on the original image
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            intercept = rho / b
            slope = -a / b

            if (np.abs(slope) < slope_thr and b!=0):
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                horizontal_lines_params.append([intercept, slope])
                cv2.line(img_horizontal_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)

    horizontal_lines_params = np.array(horizontal_lines_params)            
    # Display the final image with detected lines
    if plotting:
        show_img(img_horizontal_lines, figsize=(7, 7), title=f'{orientation} Lines')

    # --------------------- Clustering -----------------------------------------------
#     if len(horizontal_lines_params.shape) != 2:
#         return [[0,0]]
    intercepts, slopes = horizontal_lines_params[:,0], horizontal_lines_params[:,1]

    # Creazione della matrice dei dati
    data = np.column_stack((intercepts, slopes))

    # Creazione del modello DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(data)

    # Etichette dei cluster
    labels = dbscan.labels_

    # Calcolo dei centroidi e delle medie dei coefficienti angolari per ogni cluster
    unique_labels = set(labels)
    centroids = []
    average_slopes = []

    for label in unique_labels:
        if label == -1:
            continue  # Ignora i punti rumore senza cluster
        cluster_points = data[labels == label]
        centroid = np.mean(cluster_points, axis=0)
        avg_slope = np.mean(cluster_points[:, 1])
        centroids.append(centroid)
        average_slopes.append(avg_slope)

    return centroids

# ------------------------------- CHECK IF SQUARE AND AREA THR ----------------------------------------------
def approximate_to_quadrilateral(contour):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        return approx
    else: 
        return None

def find_longest_shortest_side(approx):
    approx = np.squeeze(approx)  # Riduce le dimensioni dell'array di approssimazione

    # Calcola le coordinate dei punti adiacenti
    p1 = approx
    p2 = np.roll(approx, -1, axis=0)

    # Calcola le lunghezze dei lati
    side_lengths = np.linalg.norm(p2 - p1, axis=1)

    longest_side = np.max(side_lengths)
    shortest_side = np.min(side_lengths)

    return longest_side, shortest_side

# ---------------------------------- PLOT ON PERSPECTIVE OPTIMIZATION VALUES ----------------------
def plot_on_perspective(angles, values, text=False, roundness=2, 
                        figsize=(6, 6), ax=None, 
                        x_label='x', y_label='y', title='perspective plot'):

    ta, tv = type(angles), type(values)
    if ta is not list or tv is not list:
        raise ValueError('angles and values need to be of type \'list\'')
    # Convertiamo gli angoli da gradi a radianti
    angles_rad = np.deg2rad(angles)

    # Calcoliamo le coordinate finali delle linee
    x_end = np.array(values) * np.cos(angles_rad)
    y_end = np.array(values) * np.sin(angles_rad)

    # Tracciamo il grafico
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()

    for x, y, v in zip(x_end, y_end, values):
        ax.plot([0, x], [0, y], 'ro-')
        if text:
            ax.text(x, y, f'{np.round(v, roundness)}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    L = np.max(values)
    ax.set_xlim(- L * 0.09, L * 1.01)
    ax.set_ylim(0, L * 1.1)
    ax.grid(True)

    if not ax:
        plt.show()
    
# -------------------------------SHOW SQUARES-----------------------------------------------------
def show_squares(squares, img, width=500, height=500, square_names=True, return_squares=False):
    warp_points = find_warp_points(img) # find blue points
    img_warped = warp(img, warp_points, width, height) # warp image
    images = []
    for i, contour in enumerate(squares):
        if contour is not None:
            # Trova i limiti del rettangolo di delimitazione per il contorno
            x, y, w, h = cv2.boundingRect(contour)
            # Ridimensiona il rettangolo di delimitazione alle dimensioni desiderate (300, 300)
            resized_rect = cv2.resize(img_warped[y:y+h, x:x+w], (300, 300))
            image = cv2.cvtColor(resized_rect, cv2.COLOR_BGR2RGB)
            images.append(image)
            plt.imshow(image)
            plt.axis('off')
            if square_names:
                letter, number = chr(65 + (7 - i % 8)), i // 8 + 1
                plt.title(f'{letter}{number}')
            plt.show()
        else:
            print("Nessun contorno selezionato.")
    if return_squares:
        return images

# -----------------------------------OPTIONAL: SAVE SQUARES---------------------------------------
def save_squares(images, dir_path):
    now = datetime.datetime.now()
    # Creazione della directory se non esiste già
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Salvataggio delle immagini nella directory
    for i, image in enumerate(images):
        image_path = os.path.join(dir_path, f"{now.day}_{now.hour}_{now.minute}_image_{i}.jpg")  # Percorso completo del file
        # Scambio dei canali 'red' e 'blue'
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(image_path, image_rgb)
        print(f"Image {i} saved.")
        clear_output(wait=True)
    print(f'{len(images)} images saved successfully at path {dir_path}')

# -------------------------------------------------------- YOLO DETECTION ---------------------------------
def return_detection_results(model_path, image_path, width=500, height=500):
    # Load local model
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    warp_points = find_warp_points(img) # find blue points
    img_warped = warp(img, warp_points, width, height) # warp image
    results = model(img_warped)

def middle_point(coordinate, low_point=False, low_offset=0-85):
    x_a, y_a, x_b, y_b = coordinate
    x_intermedio = (x_a + x_b) / 2
    y_intermedio = (y_a + y_b) / 2
    if low_point:
        y_intermedio = np.max([y_a, y_b]) - (np.abs(y_a - y_b)) * low_offset
    return [x_intermedio, y_intermedio]

def return_detected_points(results):
    N = len(results[0].boxes)
    points_list = []
    for i in range(N):
        points = results[0].boxes[i].data[0][:4]
        x, y = middle_point(points)
        x, y = x.item(), y.item() # from 'torch.tensor' to scalar
        points_list.append([x, y])
        
    return points_list

def remove_points_outside_quadrilateral(warp_points, output_points):
    # Converte le coordinate in array NumPy
    warp_points = np.array(warp_points, dtype=np.float32)
    output_points = np.array(output_points, dtype=np.float32)

    # Crea una maschera di regione di interesse utilizzando il quadrilatero
    mask = np.zeros_like(output_points[:, 0], dtype=np.uint8)
    cv2.fillPoly(mask, [warp_points.astype(np.int32)], 1)

    # Filtra le coordinate in output_points utilizzando la maschera
    output_points_filtered = output_points[mask.astype(bool)]

    return output_points_filtered

def group_points(points, classes, thr):
    # Converto le liste di punti e classi in array numpy per una gestione più efficiente
    points_array = points.copy()
    classes_array = classes.copy()

    # Costruisco l'albero dei punti
    kdtree = cKDTree(points_array)

    # Trovo i vicini per ogni punto
    neighbors = kdtree.query_ball_point(points_array, thr)

    # Calcolo i punti medi approssimati per ogni gruppo di vicini e aggiorno le classi corrispondenti
    grouped_points = []
    grouped_classes = []
    processed_indices = set()
    for i, group in enumerate(neighbors):
        if i not in processed_indices:
            if group:
                group_points_array = points_array[group]
                group_classes_array = classes_array[group]
                group_mean = np.mean(group_points_array, axis=0)
                grouped_points.append(group_mean)
                grouped_classes.append(group_classes_array[0])
                processed_indices.update(group)

    return np.array(grouped_points), np.array(grouped_classes)


def get_pieces(model_path, image_path, width=500, height=500, border_perc=0.95,
               low_point=False, low_offset=0.85,
               grouping=True, group_thr=30,
               plotting=False):
    # Load local model
    model = YOLO(model_path)
    img = cv2.imread(image_path)
    warp_points = find_warp_points(img) # find blue points
    img_warped, warp_matrix = warp(img, warp_points, width, height, return_matrix=True) # warp image

    results = model(img)
    N = len(results[0].boxes)
    points_list = []
    classes = []
    for i in range(N):
        points = results[0].boxes[i].data[0][:4]
        cls = results[0].boxes[i].cls[0].item()
        x, y = middle_point(points, low_point=low_point, low_offset=low_offset)
        x, y = x.item(), y.item() # from 'torch.tensor' to scalar
        points_list.append([x, y])
        classes.append(cls)
    
    # Apply warp to points found
    warped_pieces_points = cv2.perspectiveTransform(np.array([points_list]), warp_matrix)[0]


    filtered_warped_pieces_points, filtered_classes = [], []
    for i in range(N):
        x, y = warped_pieces_points[i]
        c = classes[i]
        if x <= (width - width*border_perc) or x >= width*border_perc or y <= (height - height*border_perc) or y >= height*border_perc:
            pass
        else:
            filtered_warped_pieces_points.append([x, y])
            filtered_classes.append(c)

    filtered_classes = np.array(filtered_classes)
    filtered_warped_pieces_points = np.array(filtered_warped_pieces_points)
    
    if grouping:
        filtered_warped_pieces_points, filtered_classes = group_points(filtered_warped_pieces_points, filtered_classes, thr=group_thr)
    N2 = len(filtered_warped_pieces_points)

    if plotting:
        colors = ['red', 'green', 'yellow', 'cyan', 'purple', 'lightgreen', 'pink', 'orange', 'skyblue',
                  'green', 'yellow', 'cyan', 'purple', 'lightgreen', 'pink', 'orange', 'skyblue']
        title = f'white:{len(filtered_classes[filtered_classes==0])} black:{len(filtered_classes[filtered_classes==1])}' # add to show_img() in case
        show_img(img_warped, title='')
        for i in range(N2):
            x, y = filtered_warped_pieces_points[i]
            plt.scatter(x, y, c=colors[int(filtered_classes[i])])
        plt.show()
    
    
        
    return filtered_warped_pieces_points, filtered_classes

# ---------------------------------------- SAVE PIECES CLASS AND POSITION (CHECKERS) -------------------------------
def get_piece_type(symbol):
    piece_mapping = {
        'R': chess.ROOK,
        'N': chess.KNIGHT,
        'B': chess.BISHOP,
        'K': chess.KING,
        'Q': chess.QUEEN,
        'P': chess.PAWN
    }
    
    # piece color in 'chess' syntax
    piece_color = chess.WHITE if symbol.isupper() else chess.BLACK
    
    # 'chess' piece.type
    return chess.Piece(piece_mapping[symbol.upper()], piece_color)

def get_board(pieces_points, classes, squares, board_type='chess',
              display_board=False, size_display=300):
    Chessboard = chess.Board()
    Chessboard.clear()

    
    if board_type == 'chess':
        names = ['?', 'b', 'k', 'n', 'p', 'q', 'r',
                 'B', 'K', 'N', 'P', 'Q', 'R']
    elif board_type == 'checkers':
        names=['W', 'B']
    else:
        raise ValueError('Wrong board type!')
    
    positions = []
    matrix = np.full((8, 8), '.', dtype=str)
    
    for idx_point in range(len(pieces_points)):
        first_point = pieces_points[idx_point]
        class_first_point = classes[idx_point]
        
        for i in range(64):
            r, c = 7 - i // 8, 7 - i % 8
            letter, number = chr(65 + (7 - i % 8)), i // 8 + 1
            position = f'{letter}{number}'
            # Primo elemento di squares
            first_square = squares[i]

            # Conversione del primo elemento di squares in un array di tipo np.int32
            first_square = np.array(first_square, dtype=np.int32)

            # Verifica se il primo punto è all'interno del primo contorno
            is_inside = cv2.pointPolygonTest(first_square, first_point, False) > 0
        
            if is_inside:
                piece = names[int(class_first_point)]
                matrix[r, c] = piece
                positions.append([class_first_point, position])
                Chessboard.set_piece_at(chess.parse_square(position.lower()), get_piece_type(piece))

            else:
                pass
    if display_board:
        svg_wrapper = chess.svg.board(Chessboard, size=size_display)
        display(SVG(svg_wrapper))
        
    return Chessboard, np.array(positions)
# ------------------------------ CHESS PIECES DETECTION ------------------------------------------------------------



# ==================================================================================================================
# ===================================================CLASS =========================================================
# ==================================================================================================================


class ChessDetector:
     
    # input: @ dictionary of parameters                                                             

    def __init__(self, inputs):
        
        # Processing Image
        self.blur_k = inputs["blur_k"]
        self.contrast = inputs["contrast"]
        self.canny_thr_min = inputs["canny_thr_min"]
        self.canny_thr_max = inputs["canny_thr_max"]
        self.apertureSize = inputs["apertureSize"]
        
        # Warping
        self.width = inputs["width"]
        self.height = inputs["height"]
        
        # Hough transform
        self.rho = inputs["rho"]
        self.theta_divider = inputs["theta_divider"]
        self.HT_threshold = inputs["HT_threshold"]
        self.slope_thr = inputs["slope_thr"]
        
        # Horizontal/Vertical Clustering
        self.eps = inputs["eps"]
        self.min_samples = inputs["min_samples"]
        
        # Mask Processing
        self.slope_0 = inputs["slope_0"]
        self.area_thr = inputs["area_thr"]
        self.sides_ratio = inputs["sides_ratio"]
        
        # Optimization
        self.method = 'Powell' #inputs["method"]
        
    #============================================================================================================================
    # 2.0 Return Detector Parameters
    
    def ReturnParameters(self):
        self_values = self.__dict__
        return self_values
        
    #============================================================================================================================
    # 2.1 Manual Input Parameters -> Squares Search:

    def Find_Squares(self, img, inputs, plot_detection=True, figsize=(6,6)):
        

        # Processing Image
        blur_k = inputs["blur_k"]
        contrast = inputs["contrast"]
        canny_thr_min = inputs["canny_thr_min"]
        canny_thr_max = inputs["canny_thr_max"]
        apertureSize = inputs["apertureSize"]

        # Warping
        width = inputs["width"]
        height = inputs["height"]

        # Hough transform
        rho = inputs["rho"]
        theta_divider = inputs["theta_divider"]
        HT_threshold = inputs["HT_threshold"]
        slope_thr = inputs["slope_thr"]

        # Horizontal/Vertical Clustering
        eps = inputs["eps"]
        min_samples = inputs["min_samples"]

        # Mask Processing
        slope_0 = inputs["slope_0"]
        area_thr = inputs["area_thr"]
        sides_ratio = inputs["sides_ratio"]

#         #         parameters = self.__dict__
#         blur_k, contrast, canny_thr_min, canny_thr_max, apertureSize = ordered_values[:5]
#         width, height, rho, theta_divider, HT_threshold, slope_thr = ordered_values[5:11]
#         eps, min_samples, slope_0, area_thr, sides_ratio = ordered_values[11:] #method!!

        warp_points = find_warp_points(img) # find blue points
        img_warped = warp(img, warp_points, width, height) # warp image
        blurred_image = cv2.blur(img_warped, (blur_k, blur_k)) # blur
        gray_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY) # grayscale
        offset = 127 * (1 - contrast) 
        eq_img = cv2.addWeighted(gray_image, contrast, np.zeros_like(gray_image), 0, offset) # equalize
        edges = cv2.Canny(eq_img, canny_thr_min, canny_thr_max, apertureSize=apertureSize)
        # Horizontal/Vertical 
        centroids_h = np.array(find_centroids(edges, img, 
                                              slope_thr=slope_thr,
                                              eps=eps, min_samples=min_samples))
        centroids_v = np.array(find_centroids(edges, img, 
                                              slope_thr=slope_thr,
                                              eps=eps, min_samples=min_samples, 
                                              orientation='vertical'))

        # Masks
        mask_h = np.zeros((height, width), dtype=np.uint8)
        mask_v = np.zeros((height, width), dtype=np.uint8)
        # Draw on masks
        for centroid in centroids_h:
            intercept = centroid[0]
            slope = centroid[1]
            if slope_0 > 0.5:
                slope = 0
            x1 = 0  # Coordinata x di un punto iniziale
            y1 = int(intercept)  # Coordinata y corrispondente al punto iniziale
            x2 = width  # Coordinata x di un punto finale
            y2 = int(intercept + slope * width * 2)  # Coordinata y corrispondente al punto finale
            cv2.line(mask_h, (x1, y1), (x2, y2), 255, thickness=2)
        for centroid in centroids_v:
            intercept = centroid[0]
            slope = centroid[1]
            if slope_0 > 0.5:
                slope = 0
            x1 = 0  # Coordinata x di un punto iniziale
            y1 = int(intercept)  # Coordinata y corrispondente al punto iniziale
            x2 = width  # Coordinata x di un punto finale
            y2 = int(intercept + slope * width * 2)  # Coordinata y corrispondente al punto finale
            cv2.line(mask_v, (x1, y1), (x2, y2), 255, thickness=2)
        # Rotate vertical mask
        mask_v = cv2.rotate(mask_v, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Sum masks
        final_mask_inverted = mask_h + mask_v
        final_mask = cv2.bitwise_not(final_mask_inverted)
        # Remove Intensity under 200
        final_mask[final_mask < 200] = 0

        img_with_squares = img_warped.copy()

        # Find Contouors in 'mask'
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contour over THR
        THR = area_thr
        squares = []
        for contour in contours:
            approx = approximate_to_quadrilateral(contour)
            if approx is not None:
                L, l = find_longest_shortest_side(approx)
                if (L - l) / L < sides_ratio:
                    area = cv2.contourArea(contour)
                    if area > THR:
                        squares.append(contour)
                        cv2.drawContours(img_with_squares, [contour], -1, (0, 255, 0), 2)

        if plot_detection:
            show_img(img_with_squares, title=f'Squares found: {len(squares)}', figsize=figsize)

        return squares, img_with_squares




    # ------------------------ PARAMETERS GRIDSEARCH FOR 64 SQUARES ------------------------------------
    def objective_function(self, init_values, original_img, init_keys, square_similarity=False):
        
        
    
        try:
            big_dict = self.ReturnParameters()  # Dizionario più grande
            small_dict = {key: value for (key,value) in zip(init_keys, init_values)}  # Dizionario più piccolo
            parameters = {key: value for key, value in big_dict.items() if key not in small_dict}
#             print(f'len input parameters vocab: {len(parameters.keys())}')
            parameters.update(small_dict)

            # Esegui la funzione find_squares_clustering con i parametri correnti
            sq, img_sq = self.Find_Squares(original_img, parameters, plot_detection=False)

            # Calcola la differenza tra la lunghezza di sq e il valore target 64
            diff = len(sq) - 64
            
            if square_similarity:
                areas = [cv2.contourArea(x) for x in sq]
                return diff ** 2 + (np.var(areas) ** (1/4))
            else:
                return diff ** 2
        except:
            return 100
        
        
    def Parameters_Optimization(self, img, choosen_parameters, plot_detection=True, check=False, 
                                square_similarity=False,
                                bounds=None):
        
        original_img = img.copy()
        
        init_values = list(choosen_parameters.values())
        init_keys = list(choosen_parameters.keys())
        
        if bounds is None:
            result = minimize(self.objective_function, init_values, args=(original_img, init_keys, square_similarity), method=self.method)
        else:
            bounds_tuples = [t for t in bounds.values()]
            result = minimize(self.objective_function, init_values, args=(original_img, init_keys, square_similarity), method=self.method,
                              bounds=bounds_tuples)

        # Recupera i parametri ottimali
        optimal_values = result.x
        optimal_parameters = {key: value for (key,value) in zip(init_keys, optimal_values)}
        # Build big dictionary parameters with new optimized ones
        big_dict = self.ReturnParameters()
        opt_pars = {key: value for key, value in big_dict.items() if key not in optimal_parameters}
        opt_pars.update(optimal_parameters)
        # Esegui la funzione find_squares_clustering con i parametri ottimali
        sq, img_sq = self.Find_Squares(original_img, opt_pars, plot_detection=plot_detection)

        # Verifica se la lunghezza di sq è uguale a 64
        if len(sq) == 64:
            if check:
                print("I've found the Optimal Parameters for 64 squares :)")
            return sq, img_sq, optimal_parameters
        else:
            if check:
                print("Optimal Parameters not found :(. Initial Parameters Returned")
            return None, None, [np.nan] * len(init_values)
        
    # --------------------------------------- CHESSBOARD RECONSTRUCTION --------------------------------
    def ScanBoard(self, img_path, optimization=False, board_type='chess',
                  display_board=False, size_display=300,
                  plot_detection=False, plot_squares=False,
                  return_fen=False):
        
        if board_type == 'chess':
            YOLO_trained_model = 'YOLO files/best_chess_1.pt'
        elif board_type == 'checkers':
            YOLO_trained_model = 'YOLO files/checkers_1.pt'
        else:
            raise ValueError('Choose a correct board_type ---> \'chess\' or \'checkers\'')
            
        # get inputs
        inputs = self.ReturnParameters()
        
        # Detect pieces
        pieces_points, classes = get_pieces(model_path=YOLO_trained_model,
                                                    image_path=img_path,
                                                    border_perc=0.95,
                                                    low_point=True, low_offset=0.2,
                                                    grouping=True, group_thr=45,
                                                    plotting=plot_detection)
        # Detect squares
        image = cv2.imread(img_path)
        squares, image_with_squares = self.Find_Squares(image, inputs,
                                                      plot_detection=plot_squares)
        positions, board, FEN = None, None, None
        # Get board
        try:
            board, positions = get_board(pieces_points, classes, squares,
                                             board_type=board_type,
                                             display_board=display_board, size_display=size_display)
            FEN = board.fen()
        except:
            print('Squares Detection Failed.\n(positions, board = None, None)')
            
        if return_fen:
            return board, positions, FEN

        return board, positions

















