
class MusicGenerator(object):
  """Keras Generator for Music data (Using MIXUP)"""
  def __init__(self, arg):
    super(MusicGenerator, self).__init__()
    self.arg = arg
    