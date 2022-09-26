# coding: utf-8
from IPython import get_ipython
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

"""There are 3 configuration options that you can set:
    %autoreload 0 - disables the auto-reloading. This is the default setting.

    %autoreload 1 - it will only auto-reload modules that were imported using the %aimport function (e.g %aimport my_module). It’s a good option if you want to specifically auto-reload only a selected module.

   %autoreload 2 - auto-reload """
