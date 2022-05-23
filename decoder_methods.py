DECODER_METHODS = []
UPSCALE_METHODS = ['nearest', 'bilinear', 'subpixel', 'subpixel2', 'subpixel4']

def build_deconv_methods(case2d=True):
    UPSCALE_METHODS2D = UPSCALE_METHODS
    UPSCALE_METHODS3D = [method for method in UPSCALE_METHODS if method not in ['bilinear', 'subpixel', 'subpixel2', 'subpixel4']]

    for method in UPSCALE_METHODS2D if case2d else UPSCALE_METHODS3D:
        DECODER_METHODS.extend(list(map(
            lambda transformation: method + transformation,
            ['', '_conv', '_conv2', '_conv4', '_sepconv', '_sepconv2', '_sepconv4']
        )))
        ''' 
            Стандартная реализация сепарабельной свёртки '_sepconv4' сокрашает количество каналов в 4 раза
            для увеличения разрешения представлений в 2 раза по измерению высоты и ширины.
            '_sepconv' означает, что количество каналов будет сохранено (C_out = C_in) посредством предварительного
            увеличения каналов в point-wise conv
        '''

    DECODER_METHODS.extend(['convTranspose', 'convTranspose2', 'convTranspose4'])


build_deconv_methods(case2d=False)
print(DECODER_METHODS)
