import os
import logging
from datetime import datetime

class Logger:
    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        log_filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_training.log"
        self.log_filepath = os.path.join(self.log_dir, log_filename)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(self.log_filepath)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, epoch, G_loss, D_loss, fid, mode='train'):
        log_message = f"Epoch [{epoch+1}], {mode} G_Loss: {G_loss:.4f}, {mode} D_Loss: {D_loss:.4f}, FID: {fid:.4f}"
        self.logger.info(log_message)
    
    def logd1(self, epoch, G_loss, D_loss, fid, coverage, inv_kl, mode='train'):
        log_message = f"Epoch [{epoch+1}], {mode} G_Loss: {G_loss:.4f}, {mode} D_Loss: {D_loss:.4f}, FID: {fid:.4f}, Mode Coverage: {coverage:.4f}, KL : {inv_kl:.4f}"
        self.logger.info(log_message)

    def log_initial(self, epochs, batch_size, device, img_name):
        self.logger.info(f"{img_name} parameter setting")
        log_message = f"Epoch : {epochs}, batch size : {batch_size}"
        self.logger.info(log_message)
        self.logger.info("################ Training Start ################")
    
    def log_final(self, total_epochs, final_G_loss,  final_D_loss, final_fid, mode='train'):
        self.logger.info(f"Training finished after {total_epochs} epochs.")
        self.logger.info(f"Final {mode} G_Loss: {final_G_loss:.4f}, D_Loss: {final_D_loss:.4f}, Final FID: {final_fid:.4f}")

