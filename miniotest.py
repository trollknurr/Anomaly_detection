file_path = str(model_input["path"][0])
        local_path='/tmp/' + file_path
        s3.Bucket('testdata').download_file(file_path, local_path)
        data = preprocess(local_path)
        result = self.model.predict(data)