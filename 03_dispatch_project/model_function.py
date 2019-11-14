def save_model(model, path = 'model.pkl'):
    import pickle
    F = open(path, 'wb')
    pickle.dump(model, F)
    F.close()

def load_model(path = None):
    if path == None:
        print("please input path of model")
    else:
        import pickle
        F = open(path, 'rb')
        model = pickle.load(F)
        F.close()
        return model