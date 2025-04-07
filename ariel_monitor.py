import time
import psutil
import logging
import torch

class TrainingMonitor:
    def __init__(self, logger):
        self.logger = logger
        self.start_time = None
        self.metrics = {}
    
    def start_training(self):
        self.start_time = time.time()
        self.logger.info("training", "Training started")
        self.log_system_info()
    
    def log_system_info(self):
        """Log system resource information"""
        gpu_info = {}
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "memory_used": torch.cuda.memory_allocated() / (1024**3)
            }
        
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "ram_percent": psutil.virtual_memory().percent,
            "gpu": gpu_info
        }
        
        self.logger.info("performance", f"System info: {system_info}")
    
    def log_batch(self, batch_idx, loss, lr, step, epoch):
        """Log training batch information"""
        self.logger.info("training", 
                       f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.4f}, LR: {lr:.6f}")
        
        # Log memory periodically
        if step % 50 == 0:
            self.log_system_info()
