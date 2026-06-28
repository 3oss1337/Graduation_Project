import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate {
  private let channelName = "furnishar/onnx"

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let controller = window?.rootViewController as! FlutterViewController
    let channel = FlutterMethodChannel(name: channelName, binaryMessenger: controller.binaryMessenger)

    channel.setMethodCallHandler { [weak self] call, result in
      switch call.method {
      case "hasModels":
        result(self?.hasBundledModels() ?? false)
      case "runPipeline":
        result(FlutterError(
          code: "onnx_not_configured",
          message: "iOS ONNX Runtime sessions are scaffolded but not wired. Add onnxruntime-objc or onnxruntime-c and implement MobileSAM, rembg, and MobileNetV4-small inference here.",
          details: nil
        ))
      default:
        result(FlutterMethodNotImplemented)
      }
    }

    GeneratedPluginRegistrant.register(with: self)
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  private func hasBundledModels() -> Bool {
    let names = [
      "mobile_sam_encoder",
      "mobile_sam_decoder",
      "rembg",
      "mobilenetv4_small"
    ]
    return names.allSatisfy { Bundle.main.path(forResource: $0, ofType: "onnx") != nil }
  }
}
