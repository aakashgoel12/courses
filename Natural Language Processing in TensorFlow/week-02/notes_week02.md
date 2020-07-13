1) Tensorflow data services (tfds)
	
	pip install -q tensorflow-datasets
	
2) http://ai.stanford.edu/~amaas/data/

3) http://ai.stanford.edu/~amaas/data/sentiment/

	50,000 movie reviews which are classified as positive of negative. 

4) Eager execution is enabled in TensorFlow 2.0 but not in TensorFlow 1.0. So, use this command tf.enable_eager_execution()

5) Embeddings can be visualized using https://projector.tensorflow.org/ 
Need to upload two files : meta.tsv (words) and vecs.tsv (embeddings)

6) Some points for Google Colab
	A) Download file from colab using below code:
		from google.colab import files
		files.download('vecs.tsv')

7) Once you have Embedding Layer which is 2-D i.e. (max_len words in sentence, embedding dimension) there are two ways you can flatten it:

	A) tf.keras.layers.GlobalAveragePooling1D() --> o/p size (embedding dimension) --> quite fast as relative to flatten()
	B) tf.keras.layers.flatten() --> o/p size (max_len words in sentence*embedding dimension)

8) In Model Summary, 
	A) No. of params in Embedding Layer is = Vocab size * Embedding dimension
	
	B) Loss curve shouldn't be sharp, it should be flatten.
	See before tweak of params
	[!before_tweak.png]
	
	After tweak of some params, like reduce in max_len and embedding dimension, loss curve flatten. However, accuracy decreases.
	
	[!after_tweak.png]

9) Tensorflow Links:
	A) https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder

	B) https://www.tensorflow.org/tutorials/text/transformer

	C) https://www.tensorflow.org/tutorials/text/nmt_with_attention

	D) https://github.com/tensorflow/nmt

10) sparse_categorical_crossentropy Vs categorical_crossentropy

### if your targets are integers, use sparse_categorical_crossentropy else
### If your targets are one-hot encoded, use categorical_crossentropy
