// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Mossformer2MLXSwift",
    platforms: [
        .macOS("13.3"),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "Mossformer2MLXSwift",
            targets: ["Mossformer2MLXSwift"]),
        .executable(
            name: "generate",
            targets: ["Generate"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", exact: "1.0.0"),
        .package(url: "https://github.com/starkdmi/SwiftAudio", exact: "1.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0")
    ],
    targets: [
        .target(
            name: "Mossformer2MLXSwift",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio")
            ]),
        .executableTarget(
            name: "Generate",
            dependencies: [
                "Mossformer2MLXSwift",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "AudioUtils", package: "SwiftAudio"),
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "ArgumentParser", package: "swift-argument-parser")
            ],
            path: "Sources/Generate"
        )
    ]
)
