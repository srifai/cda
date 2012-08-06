import PIL, PIL.Image
import numpy
import numpy as np
import theano
import os
def scale_to_unit_interval(ndar,eps=1e-8,scaling="centered"):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  if scaling=="centered":
    absmin = abs(ndar.min())
    absmax = abs(ndar.max())
    rescale = max([absmin,absmax])
    ndarc = ndar.copy()
    return ((ndarc / rescale)+1.)/2.
  else:
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max()+eps)
    return ndar


def tile_raster_images(X, img_shape, tile_shape,tile_spacing = (0,0),
            scale_rows_to_unit_interval = True, output_pixel_vals = True,scaling="centered"):
  """                                                                                                                                                                                                            
  Only works if input is float (due to PIL bug)                                                                                                                                                                  
  if greyscale image, feed it a matrix                                                                                                                                                                           
  if rgb, feed it 3 matrix (R,G,B)                                                                                                                                                                               
  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2


  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 3
      if output_pixel_vals:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 3), dtype='uint8')
      else:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 3), dtype='float32')

      if output_pixel_vals:
          channel_defaults = [0,0,0]
      else:
          channel_defaults = [0.,0.,0.]

      for i in xrange(3):
          if X[i] is None:
            out_array[:,:,i] = numpy.zeros(out_shape,
                                           dtype='uint8' if output_pixel_vals else out_array.dtype
                                           )+channel_defaults[i]
          else:
            out_array[:,:,i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      H, W = img_shape
      Hs, Ws = tile_spacing
      out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in xrange(tile_shape[0]):
          for tile_col in xrange(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape),scaling=scaling)
                  else:
                    this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  out_array[
                      tile_row * (H+Hs):tile_row*(H+Hs)+H,
                      tile_col * (W+Ws):tile_col*(W+Ws)+W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array


def view_data(data,name,nb_filters=-1,nb_horizontal=-1,nb_vertical=-1):

    if nb_filters==-1:
        nb_filters=data.shape[0]
    if nb_horizontal == -1:
        nb_horizontal =int( np.sqrt(nb_filters) )
    if nb_vertical == -1:
        nb_vertical = int( nb_filters/nb_horizontal )
    temp_size=data.shape[1]
    temp_size=numpy.ceil(numpy.sqrt(temp_size))
    matrix=numpy.zeros((nb_filters,temp_size*temp_size),dtype=theano.config.floatX)
    matrix[0:nb_filters,0:data.shape[1]]=data[0:nb_filters,]
    image_size=(temp_size,temp_size)

    image_data=tile_raster_images(matrix,image_size,(nb_horizontal,nb_vertical),tile_spacing = (2,2),scale_rows_to_unit_interval=True,scaling='patate')
    image=PIL.Image.fromarray(image_data)
    #name='data_image'                                                                                                                                                                                           
    file_namec = name + '.png'
    image.save(file_namec)

    return image

def view_rgb_data(data,name,nb_filters=-1):

    if nb_filters==-1:
        nb_filters=data.shape[0]
    nb_horizontal =int( np.sqrt(nb_filters) )
    nb_vertical = int( nb_filters/nb_horizontal )
    temp_size=data.shape[1]/3
    temp_size=numpy.ceil(numpy.sqrt(temp_size))
    matrix=numpy.zeros((nb_filters,temp_size*temp_size*3),dtype='float32')
    matrix[0:nb_filters,0:data.shape[1]]=data[0:nb_filters,]
    image_size=(temp_size,temp_size)
    size=image_size[0]*image_size[1]
    matrix = matrix.reshape((nb_filters,3,temp_size*temp_size))
    R=matrix[:,0,:]
    G=matrix[:,1,:]
    B=matrix[:,2,:]
    image_data=tile_raster_images((R,G,B),image_size,(nb_horizontal,nb_vertical),tile_spacing = (1,1),scale_rows_to_unit_interval=True,scaling='centered')
    image=PIL.Image.fromarray(image_data)
    file_namec = name + '.png'
    image.save(file_namec)

    return image
