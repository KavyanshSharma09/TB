from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from Mainapp.views import *
urlpatterns = [
    path("admin/", admin.site.urls),
    path("",homepage,name='homepage'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
