import torch
import os

class EarlyStopping:
    def __init__(self, patience:int, delta:float, device:str, mode='min', verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): If True, prints a message for each improvement.
            delta (float): Minimum change to qualify as improvement.
            mode (str): 'min' for loss, 'max' for accuracy, etc.
        """
        self.patience = patience
        self.delta = delta
        self.device = device
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False



    def __call__(self,model:any, checkpoint_path:str,optimizer:any, epochs:int, step:int,val_loss:float)->None:
        score = -val_loss if self.mode == 'min' else val_loss #(-)a higher score always means improvement

        if self.best_score is None:  #first call
            self.best_score = score
            self.save_checkpoint(model,checkpoint_path,optimizer, epochs, step,self.best_score)
        elif score < self.best_score + self.delta:  #no improvement
            self.counter += 1
            if self.verbose:
                print(f"INFO(Early stopping): EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("INFO(Early stopping): Early stopping triggered.")
                self.early_stop = True
        else: #score > best_score indicates improvement
            self.best_score = score #i
            self.save_checkpoint(model,checkpoint_path,optimizer, epochs, step,self.best_score)
            self.counter = 0

    def save_checkpoint(self, chatGPT_model:any, checkpoint_path:str, optimizer:any, epochs:int, step:int, val_loss:float,dispatcher="EarlyStopping")->None:
        #torch.save(model.state_dict(), self.save_path)
        checkpoint = {
            'model_state_dict': chatGPT_model.state_dict(),  # for loading the model outside of DDP or for inference.
            'model_config': chatGPT_model.config,
            'optimizer_state_dict': optimizer.state_dict(),
            'epochs': epochs,
            'step': step,
            'val_loss': val_loss
        }
        #you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)
        if self.verbose:
            print(f"INFO({dispatcher}):Improved model saved to {checkpoint_path}")


    def loadModel(self, model:any, checkpoint_path:str, optimizer:any, mode:str)->any:
        #print("loading model: ", PATH)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path,map_location=self.device ,weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epochs']
            val_loss = checkpoint['val_loss']
            print("INFO(Checkpoint): Loaded model : epoch {} - val_loss {}: ".format(epoch,val_loss))
            model.train() if mode == 'train' else model.eval()  #set model
        return model