def deco(msg):
  def inner_deco(func):
    def new_func(*args, **kwargs):
      print(f'[log] run function {func.__name__}')
      rlt = func(*args, **kwargs)
      print(f'[log] {msg}')
      return rlt
    return new_func
  return inner_deco