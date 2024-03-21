import pickle
from io import BytesIO
from Models import Model

class BaseModelManager:
    __baseModel = None

    @classmethod
    def set_base_model(cls, model: Model):
        """
        Sets the base model for the manager, ensuring Singleton behavior.
        
        Parameters:
            model (Model): The model to be set as the base model.
            
        Raises:
            TypeError: If the base model has already been initialized.
        """
        if cls.__baseModel is None:
            cls.__baseModel = model
        else:
            raise TypeError("The Base Model has already been initialized")

    @classmethod
    def get_instance(cls) -> Model:
        """
        Returns the instance of the base model, ensuring Singleton access.
        
        Returns:
            The base model instance.
            
        Raises:
            TypeError: If the base model has not been initialized yet.
        """
        if cls.__baseModel is None:
            raise TypeError("The Base Model has not been initialized yet")
        return cls.__baseModel

    @classmethod
    def clone_base_model(cls) -> Model:
        """
        Creates and returns a deep clone of the base model, following the Prototype pattern.
        
        This method uses serialization and deserialization to clone complex model attributes,
        allowing for independent modification of the cloned model.
        
        Returns:
            A cloned instance of the base model.

        Raises:
            TypeError: If the base model has not been initialized yet.
        """
        if cls.__baseModel is None:
            raise TypeError("The Base Model has not been initialized and cannot be cloned")
        else:
            cloned_model = type(cls.__baseModel)()
            # Serialize and deserialize the entire base model to create a deep clone.
            if hasattr(cls.__baseModel, 'model') and cls.__baseModel.model is not None:
                buffer = BytesIO()
                pickle.dump(cls.__baseModel.model, buffer)
                buffer.seek(0)
                cloned_model.model = pickle.load(buffer)
                cloned_model.model_class = cls.__baseModel.model_class
                cloned_model.pickled_model = True
            else:
                for attribute, value in vars(cls.__baseModel).items():
                    setattr(cloned_model, attribute, value)
            
            return cloned_model
