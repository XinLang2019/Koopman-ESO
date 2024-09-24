import numpy as np

class KoopmanSystem:
    def __init__(self, NK=49, nX=14, nU=7):
        self.nX = nX
        self.__G = np.zeros((NK, NK))
        self.__A = np.zeros((NK, NK))
        self.__K = np.zeros((NK, NK))
        self.__Ktilde = np.zeros((nX, NK))
    
    def fk(self, x, u):
        pass
    
    def gradStep(self, dataIn, cdataIn, dataOut):
        phix = self.fk(dataIn, cdataIn)
        phixpo = self.fk(dataOut, cdataIn)

        self.__G += np.outer(phix, phix)
        self.__A += np.outer(phix, phixpo)

        try:
            self.__K = np.pinv(self.__G) @ self.__A
        except np.linalg.LinAlgError:
            print("CAUGHT THAT ERROR")

        k_temp = self.__K.T
        self.__Ktilde = k_temp[:self.nX, :]