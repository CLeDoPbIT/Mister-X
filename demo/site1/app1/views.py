
import logging

from django.http import HttpResponse, HttpResponseBadRequest, JsonResponse
from django.views.generic import TemplateView
from django.shortcuts import redirect, render
from .models import Document
from .forms import DocumentForm

from . import plots

logger = logging.getLogger(__name__)


class IndexView(TemplateView):
    template_name = "index.html"


class Plot1DView(TemplateView):
    template_name = "plot.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(Plot1DView, self).get_context_data(**kwargs)
        context['plot'] = plots.plot1d()
        return context


def my_view(request):
    print(f"Great! You're using Python 3.6+. If you fail here, use the right version.")
    message = 'Upload test.csv'
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile=request.FILES['docfile'])
            newdoc.save()

            # Redirect to the document list after POST
            return redirect('app1/plot1d/')
        else:
            message = 'The form is not valid. Fix the following error:'
    else:
        form = DocumentForm()  # An empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()

    # Render list page with the documents and the form
    context = {'documents': documents, 'form': form, 'message': message}
    return render(request, 'list.html', context)



def plot1d_multiple_ajax(request, n):
    """
    Only handles AJAX queries
    """
    logger.debug("Plotting {} plots.".format(n))
    return HttpResponse(plots.plot1d_multiple(int(n)))


class PlotIqView(TemplateView):
    template_name = "plot_fit.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(PlotIqView, self).get_context_data(**kwargs)
        context['plot'] = plots.plotIq()
        return context


class PlotLiveView(TemplateView):
    template_name = "plot_live.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(PlotLiveView, self).get_context_data(**kwargs)
        context['plot'] = plots.plotLive()
        return context


def plot_live_update(request):
    '''
    Handle ajax call to update the live plot
    '''
    if request.is_ajax():
        logger.debug("Live plot updated...")
        data = plots.live_plot_get_data_serialized()
        # In order to allow non-dict objects to be serialized set the safe
        # parameter to False
        return JsonResponse([data], safe=False)
    else:
        return HttpResponseBadRequest()

class Plot3DScatterView(TemplateView):
    template_name = "plot.html"

    def get_context_data(self, **kwargs):
        # Call the base implementation first to get a context
        context = super(Plot3DScatterView, self).get_context_data(**kwargs)
        context['plot'] = plots.plot3D_scatter
        return context