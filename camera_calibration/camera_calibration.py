import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

class CameraSystem:
    def __init__(self, checkerboard=(6,9), square_size=1.0):
        self.checkerboard = checkerboard
        self.square_size = square_size
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
    def calibrate_single(self, image_path_pattern):
        """Tek kamera kalibrasyonu"""
        # Object points (3D)
        objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        images = glob.glob(image_path_pattern)
        
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # tahtanın koselerini bul
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard, None)
            
            if ret:
                objpoints.append(objp)
                
                # Kosa hassasiyeti arttirma 
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners_refined)
        
        # Kamera kalibrasyonu
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        print("Kamera Kalibrasyonu Tamamlandı!")
        print("Kamera Matrisi:\n", self.camera_matrix) #icsel parametreler
        print("Distorsiyon Katsayıları:", self.dist_coeffs.ravel()) #distorsiyon katsayilari
        
        return ret
    
    def calibrate_stereo(self, other_cam, left_images_pattern, right_images_pattern):
        """Stereo kamera kalibrasyonu"""
        # Object points
        objp = np.zeros((self.checkerboard[0] * self.checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard[0], 0:self.checkerboard[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        
        left_images = sorted(glob.glob(left_images_pattern))
        right_images = sorted(glob.glob(right_images_pattern))
        
        for left_fname, right_fname in zip(left_images, right_images):
            left_img = cv2.imread(left_fname)
            right_img = cv2.imread(right_fname)
            
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # Find corners in both images
            ret_left, corners_left = cv2.findChessboardCorners(left_gray, self.checkerboard, None)
            ret_right, corners_right = cv2.findChessboardCorners(right_gray, self.checkerboard, None)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_left_refined = cv2.cornerSubPix(left_gray, corners_left, (11,11), (-1,-1), criteria)
                corners_right_refined = cv2.cornerSubPix(right_gray, corners_right, (11,11), (-1,-1), criteria)
                
                imgpoints_left.append(corners_left_refined)
                imgpoints_right.append(corners_right_refined)
        
        # Stereo kalibrasyon
        ret, self.camera_matrix, self.dist_coeffs, other_cam.camera_matrix, other_cam.dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix, self.dist_coeffs,
            other_cam.camera_matrix, other_cam.dist_coeffs,
            left_gray.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        print("Stereo Kalibrasyon Tamamlandı!")
        print("Döndürme Matrisi (R):\n", R)
        print("Öteleme Vektörü (T):", T.ravel())
        print("Esansiyel Matris (E):\n", E)
        print("Temel Matris (F):\n", F)
        
        # Stereo rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.camera_matrix, self.dist_coeffs,
            other_cam.camera_matrix, other_cam.dist_coeffs,
            left_gray.shape[::-1], R, T
        )
        
        return {
            'R': R, 'T': T, 'E': E, 'F': F,
            'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2, 'Q': Q
        }
    
    def compute_homography(self, src_points, dst_points):
        """Homografi hesaplama"""
        H, status = cv2.findHomography(src_points, dst_points)
        print("Homografi Matrisi:\n", H)
        return H, status
    
    def apply_homography(self, img, H, output_size):
        """Homografi uygulama"""
        return cv2.warpPerspective(img, H, output_size)
    
    def compute_disparity(self, left_img, right_img):
        """Derinlik haritası için disparity hesaplama"""
        # Stereo eşleştirme için parametreler
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=11,
            P1=8*3*11**2,
            P2=32*3*11**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0
        return disparity
    
    def disparity_to_depth(self, disparity, stereo_params):
        """Disparity'den derinlik haritası oluşturma"""
        if disparity is None:
            return None
            
        # Q matrisini kullanarak disparity'yi derinliğe dönüştür
        depth = cv2.reprojectImageTo3D(disparity, stereo_params['Q'])
        depth_map = depth[:,:,2]  # Z bileşeni derinlik bilgisidir
        
        # Geçersiz değerleri filtrele
        depth_map[disparity <= 0] = 0
        
        return depth_map
    
    def project_3d_points(self, img, points_3d):
        """3D noktaları 2D'ye projeksiyon"""
        if self.camera_matrix is None:
            raise ValueError("Önce kamera kalibrasyonu yapılmalı!")
        
        # 3D noktaları 2D'ye projeksiyon
        points_2d, _ = cv2.projectPoints(
            points_3d, self.rvecs[0], self.tvecs[0], 
            self.camera_matrix, self.dist_coeffs
        )
        points_2d = points_2d.reshape(-1, 2).astype(int)
        
        # Küp çizimi
        img_copy = img.copy()
        
        # Alt yüzey (taban)
        for i in range(4):
            cv2.line(img_copy, tuple(points_2d[i]), tuple(points_2d[(i+1)%4]), (0,255,0), 2)
        
        # Üst yüzey
        for i in range(4):
            cv2.line(img_copy, tuple(points_2d[i+4]), tuple(points_2d[4+(i+1)%4]), (0,255,0), 2)
        
        # Dikey kenarlar
        for i in range(4):
            cv2.line(img_copy, tuple(points_2d[i]), tuple(points_2d[i+4]), (0,255,0), 2)
        
        return img_copy

# Ana uygulama
def main():
    # 1. TEK KAMERA KALİBRASYONU
    print("=== TEK KAMERA KALİBRASYONU ===")
    cam = CameraSystem(checkerboard=(6,9), square_size=1.0)
    cam.calibrate_single("calibration_images/*.jpg")
    
    # 2. 3D NESNE PROJEKSİYONU
    print("\n=== 3D NESNE PROJEKSİYONU ===")
    cube_points = np.float32([
        [0,0,0],[1,0,0],[1,1,0],[0,1,0],   # taban
        [0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]  # üst
    ])
    
    img = cv2.imread("calibration_images/checkerboard0-200x155.jpg")
    img_proj = cam.project_3d_points(img, cube_points)
    
    cv2.imshow("3D Cube Projection", img_proj)
    cv2.waitKey(0)
    
    # 3. HOMOGRATİ HESAPLAMA
    print("\n=== HOMOGRATİ HESAPLAMA ===")
    src_pts = np.float32([[0,0], [1,0], [1,1], [0,1]])
    dst_pts = np.float32([[10,10], [100,10], [80,80], [20,70]])
    
    H, status = cam.compute_homography(src_pts, dst_pts)
    
    # 4. STEREO GÖRÜŞ 
    print("\n=== STEREO GÖRÜŞ ===")
    cam_left = CameraSystem(checkerboard=(7,11), square_size=1.0)
    cam_right = CameraSystem(checkerboard=(7,11), square_size=1.0)
    
    # Stereo kalibrasyon (gerçek uygulamada stereo görüntü çiftleri gerekli)
    try:
        stereo_params = cam_left.calibrate_stereo(cam_right, "left_images/*.png", "right_images/*.png")
    except:
        print("Stereo görüntüler bulunamadı, simüle ediliyor...")
        # Simüle stereo parametreler
        stereo_params = {
            'Q': np.array([[1, 0, 0, -320],
                          [0, 1, 0, -240], 
                          [0, 0, 0, 500],
                          [0, 0, 1/0.05, 0]])
        }
    
    # 5. DERİNLİK HARİTASI OLUŞTURMA
    print("\n=== DERİNLİK HARİTASI ===")
    # Örnek stereo görüntüler oluştur
    left_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    right_img = np.roll(left_img, 5, axis=1)  # Sağa kaydırılmış görüntü
    
    disparity = cam_left.compute_disparity(left_img, right_img)
    depth_map = cam_left.disparity_to_depth(disparity, stereo_params)
    
    # Görselleştirme
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_proj, cv2.COLOR_BGR2RGB))
    plt.title('3D Küp Projeksiyonu')
    
    plt.subplot(132)
    if disparity is not None:
        plt.imshow(disparity, cmap='jet')
        plt.title('Disparity Haritası')
    
    plt.subplot(133)
    if depth_map is not None:
        plt.imshow(depth_map, cmap='jet')
        plt.title('Derinlik Haritası')
    
    plt.tight_layout()
    plt.show()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()