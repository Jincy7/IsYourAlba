from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^test/', views.test, name='test'),
    url(r'^test_EI/(?P<username>[ㄱ-ㅣ가-힣]+)', views.test_EI, name='test_EI'),
    url(r'^test_SN/(?P<username>[ㄱ-ㅣ가-힣]+)', views.test_SN, name='test_SN'),
    url(r'^test_TF/(?P<username>[ㄱ-ㅣ가-힣]+)', views.test_TF, name='test_TF'),
    url(r'^test_JP/(?P<username>[ㄱ-ㅣ가-힣]+)', views.test_JP, name='test_JP'),
    url(r'^', views.index, name='index'),

]
