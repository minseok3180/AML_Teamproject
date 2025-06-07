import os
import logging
from datetime import datetime

class Logger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 로그 파일명: 현재 날짜로 생성
        log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_training.log"
        self.log_filepath = os.path.join(self.log_dir, log_filename)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # 파일 핸들러
        file_handler = logging.FileHandler(self.log_filepath)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # 로거에 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, epoch, G_loss, D_loss, fid, mode='train'):
        """로그를 남기는 함수 (Epoch마다 loss와 FID 기록)"""
        log_message = f"Epoch [{epoch}], {mode} G_Loss: {G_loss:.4f}, {mode} D_Loss: {D_loss:.4f}, FID: {fid:.4f}"
        self.logger.info(log_message)
    
    def log_final(self, total_epochs, final_G_loss,  final_D_loss, final_fid, mode='train'):
        """훈련 종료 후 최종 결과 로깅"""
        self.logger.info(f"Training finished after {total_epochs} epochs.")
        self.logger.info(f"Final {mode} G_Loss: {final_G_loss:.4f}, D_Loss: {final_D_loss:.4f}, Final FID: {final_fid:.4f}")

