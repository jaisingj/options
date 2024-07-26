import streamlit as st
import altair as alt
import pandas as pd
import time
import datetime as dt
import uuid
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import yahoo_fin.stock_info as si
import seaborn as sb
import plotly.express as px
import plotly.graph_objs as go
import json
import base64
import tqdm
import pandas_ta as ta
import io
import certifi
import locale
import warnings
import requests
import re
import tweepy
import time
import config
import nltk
import xlsxwriter
from pathlib import Path
#from vaderSentiment import SentimentIntensityAnalyzer
from pandas_datareader import data as pdr
from jinja2 import Environment, select_autoescape, FileSystemLoader
from pytz import timezone
from pandas.tseries.offsets import BDay
from functions import apply_custom_css, custom_css, clear_multi, get_color, get_float_value,  get_info,get_stock_industry, calculate_tier, color_cells, color_tiers, hint, apply_custom_css, generate_charts,create_download_link, get_news_yahoo, score_news,to_excel, simulate_future_value, display_stock_info , get_trade_open, get_table_download_link, display_current_price,get_stock_info, get_stock_industry,  get_last_price
from datetime import datetime, timedelta
from stock_utils import display_stock_info
from stock_utils import display_company_description
from datetime import date
from dateutil.parser import parse
from scipy.stats import iqr
from ta import add_all_ta_features
from plotly.subplots import make_subplots
from pandas_datareader import data as pdr
from pandas.tseries.offsets import BDay
from ta.trend import MACD
from ta.momentum import StochasticOscillator
from plotly.subplots import make_subplots
from urllib.request import urlopen
from IPython.display import display
from typing import Dict
from st_on_hover_tabs import on_hover_tabs
from matplotlib.backends.backend_agg import FigureCanvasAgg
from plotly import graph_objs as go
from streamlit import components
from millify import millify
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from yahoo_fin import stock_info as si
from yahoo_fin import options
from yahoo_fin import news
from datetime import datetime, timedelta 
from yahoo_fin import news
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from bs4 import BeautifulSoup
from dateutil.parser import parse
from xlsxwriter import workbook

