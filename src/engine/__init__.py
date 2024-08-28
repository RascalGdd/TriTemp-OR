from .evaluator_or import or_evaluate

def hoi_evaluator(args, model, criterion, postprocessors, data_loader, device, thr=0):
    return or_evaluate(model, postprocessors, data_loader, device, thr, args)
